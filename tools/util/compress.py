import torch

from runtime.compress.compression_policy import CompressionPolicy, LayerCompressionSpecification, load_policy
from runtime.compress.torch_compress.torch_adapters import TorchCompressAdapter
from runtime.evaluation.tvm_evaluator import TvmMapper, TvmConfig
from runtime.model.torch_model import TorchModelFactory, TorchExecutableModel


def get_compression_params(compression_spec: LayerCompressionSpecification):
    param_strings = []
    for param in reversed(compression_spec.compression_parameters):
        param_strings.append(
            f"{param.parameter_key}: c = {param.compression_ratio:3f} d = {param.target_discrete} ({param.reference})")
    return param_strings


def print_policy(policy: CompressionPolicy):
    policy = policy.get_clean_policy()
    idx = 0
    for key, layer_compression in policy.layers.items():
        print(f"{key}:")
        for method_key, method in layer_compression.items():
            print(f" - {method_key}: {get_compression_params(method)}")


def construct_adapter(model, data_provider, device,
                      enabled_methods=tuple(["p-conv", "p-lin", "q-fp32", "q-int8", "q-mixed"])):
    model_factory = TorchModelFactory(torch_model=model,
                                      batch_input_shape=data_provider.batch_input_shape,
                                      target_device=device,
                                      frozen_layers=dict())
    model_reference = model_factory.get_reference_model()
    compress_adapter = TorchCompressAdapter(model_reference=model_reference,
                                            model_factory=model_factory,
                                            enabled_methods=enabled_methods,
                                            no_discretization=True)
    return compress_adapter, model_reference


def compress_model(model: torch.nn.Module, policy_path, data_provider, device,
                   enabled_methods=tuple(["p-conv", "p-lin", "q-fp32", "q-int8", "q-mixed"])):
    executable_model, policy = _compress_to_executable(data_provider, device, model, policy_path, enabled_methods)
    if isinstance(executable_model, TorchExecutableModel):
        print("Applied compression policy to model:")
        print_policy(policy)
        return executable_model.pytorch_model, executable_model.compression_protocol
    raise Exception("Misconfiguration: expected a pytorch model")


def compress_and_map_to_relay(model: torch.nn.Module, policy_path, data_provider, device):
    executable_model, policy = _compress_to_executable(data_provider, device, model, policy_path)
    tvm_mapper = TvmMapper(data_provider, TvmConfig())

    return tvm_mapper.to_tvm_model(executable_model)


def map_to_relay(model: torch.nn.Module, data_provider, device):
    model_factory = TorchModelFactory(torch_model=model,
                                      batch_input_shape=data_provider.batch_input_shape,
                                      target_device=device,
                                      frozen_layers=dict())
    executable_model = model_factory.to_executable_model(model_factory.get_reference_model())
    tvm_mapper = TvmMapper(data_provider, TvmConfig())

    return tvm_mapper.to_tvm_model(executable_model)


def _compress_to_executable(data_provider, device, model, policy_path,
                            enabled_methods=tuple(["p-conv", "p-lin", "q-fp32", "q-int8", "q-mixed"])):
    policy = load_policy(policy_path)
    compression_adapter, model_ref = construct_adapter(model, data_provider, device, enabled_methods)
    _, executable_model = compression_adapter.compress(policy, model_ref)
    return executable_model, policy
