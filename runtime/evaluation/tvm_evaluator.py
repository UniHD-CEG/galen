import copy
import os
import time

import torch
import tvm
import tvm.relay as relay
from tqdm import tqdm
from tvm import autotvm
from tvm.contrib import graph_executor
from tvm.contrib.utils import tempdir

from runtime.data.data_provider import ADataProvider
from runtime.evaluation.evaluator import ALatencyEvaluator
from runtime.evaluation.tvm_qmapper import QuantizationOperatorMapper
from runtime.model.torch_model import TorchExecutableModel


class TvmConfig:
    def __init__(self, **kwargs):
        tvm_target_string = kwargs.pop('tvm_target', 'tvm.target.arm_cpu("rasp4b")')
        results = {}
        exec(f"target = {tvm_target_string}", globals(), results)
        self.target = results["target"]
        self.device_key = kwargs.pop("tvm_device_key", "pi-cluster-head-pi4b")
        self.rpc_ip = kwargs.pop("tvm_rpc_ip", "localhost")
        self.rpc_port = int(kwargs.pop("tvm_rpc_port", 9000))
        self.rpc_timeout = int(kwargs.pop("tvm_rpc_timout", 120))
        self.enable_tophub = kwargs.pop("enable_tophub", "False") == "True"
        self.unipolar = kwargs.pop("bitserial_unipolar", "False") == "True"


class TvmMapper:

    def __init__(self, data_provider: ADataProvider, config: TvmConfig):
        self._data_provider = data_provider
        self._cfg = config

    def to_tvm_model(self, executable_model: TorchExecutableModel):
        batch_shape = self._data_provider.batch_input_shape
        shape = tuple([1] + [el for el in batch_shape[1:]])

        copied_model = copy.deepcopy(executable_model)
        # clean up hook contexts before map -> hooks will add unnecessary relay ops
        copied_model.remove_all_contexts()

        sample_x = torch.randn(shape).to(copied_model.target_device)
        model = copied_model.pytorch_model.eval()
        scripted_model = torch.jit.trace(model, sample_x).eval()

        input_name = "input0"
        input_info = [(("%s" % input_name), shape)]

        if copied_model.applied_policy:
            q_mapper = QuantizationOperatorMapper(copied_model.applied_policy, unipolar=self._cfg.unipolar)
            custom_mapper = {
                "aten::_convolution": q_mapper.convolution,
                "aten::linear": q_mapper.linear
            }
        else:
            custom_mapper = dict()
        return relay.frontend.from_pytorch(scripted_model, input_info, custom_convert_map=custom_mapper), input_name


class TvmLatencyEvaluator(ALatencyEvaluator):

    def __init__(self, data_provider: ADataProvider, tvm_config=TvmConfig()):
        self._data_provider = data_provider
        self._cfg = tvm_config
        self._tvm_mapper = TvmMapper(self._data_provider, self._cfg)
        self._verify_tophub_state(self._cfg.enable_tophub)

    def measure_latency(self, executable_model: TorchExecutableModel) -> \
            dict[str, float]:
        bench_result = self._perform_latency_benchmark(executable_model)

        return {
            "lat": bench_result.mean
        }

    def _perform_latency_benchmark(self, executable_model):
        start_map = time.time()
        (tvm_mod, tvm_params), tvm_model_input_name = self._tvm_mapper.to_tvm_model(executable_model)
        start_compile = time.time()
        tvm_remote_device, tvm_remote_model, start_upload = self._compile_and_upload(tvm_mod, tvm_params)
        start_exec = time.time()

        tvm_execution_module = graph_executor.GraphModule(tvm_remote_model['default'](tvm_remote_device))

        img_array = self._data_provider.get_random_tensor_with_input_shape()[1, :].unsqueeze(0).cpu().detach().numpy()
        tvm_array = tvm.nd.array(img_array.astype('float32'))
        tvm_execution_module.set_input(tvm_model_input_name, tvm_array)
        tvm_execution_module.set_input(**tvm_params)

        bench_result = tvm_execution_module.benchmark(tvm_remote_device, number=5, repeat=3)
        completed = time.time()
        tqdm.write(
            f"map: {start_compile - start_map:.2f}s - compile: {start_upload - start_compile:.2f}s - upload: {start_exec - start_upload:.2f}s - exec: {completed - start_exec:.2f}s")
        return bench_result

    def _compile_and_upload(self, tvm_mod, tvm_param):
        with tvm.transform.PassContext(opt_level=3):
            compiled_model = relay.build(tvm_mod, params=tvm_param, target=self._cfg.target)
        start_upload = time.time()
        tvm_remote_model, tvm_remote_device = self._upload_model(compiled_model)
        return tvm_remote_device, tvm_remote_model, start_upload

    def _upload_model(self, compiled_model):
        # save it on temporary path
        tmp = tempdir()
        model_name = 'net.tar'
        model_path = tmp.relpath(model_name)
        compiled_model.export_library(model_path)

        # establish RPC connection
        remote = autotvm.measure.request_remote(
            self._cfg.device_key,
            self._cfg.rpc_ip,
            self._cfg.rpc_port,
            priority=10)

        # upload model
        remote.upload(model_path)

        # load model, this is the path on the remote device
        remote_handle = remote.load_module(model_name)

        # set context
        remote_device = remote.cpu(0)

        return remote_handle, remote_device

    @staticmethod
    def _verify_tophub_state(enable_tophub: bool):
        if enable_tophub:
            os.environ.pop("TOPHUB_LOCATION", None)
        else:
            os.environ["TOPHUB_LOCATION"] = "NONE"
            # not safe for concurrent renames but should be okay
            root_path = autotvm.tophub.AUTOTVM_TOPHUB_ROOT_PATH
            if root_path.exists():
                root_path.replace(str(root_path.absolute()) + "_tmp")
