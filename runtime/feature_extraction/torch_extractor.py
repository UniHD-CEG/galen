import numpy as np
import torch.nn
from fvcore.nn import FlopCountAnalysis

from runtime.compress.compression_policy import CompressionPolicy, LayerCompressionSpecification
from runtime.compress.torch_compress.torch_executors import TorchMixedQuantizationExecutor, QuantizationContext
from runtime.data.data_provider import ADataProvider
from runtime.feature_extraction.feature_extractor import AMetricExtractor, AFeatureExtractor
from runtime.model.torch_model import TorchExecutableModel
from runtime.sensitivity.sensitivity_analysis import SensitivityAnalysis


class TorchMACsBOPsExtractor(AMetricExtractor):

    def __init__(self, sample_batch):
        self._sample_batch_input = sample_batch
        self._batch_size = sample_batch[0].shape[0]
        self._executable_model = None
        self._layer_macs = None

    def __call__(self, executable_model: TorchExecutableModel):
        self._executable_model = executable_model
        return self

    def __enter__(self):
        assert self._executable_model is not None
        # the used library computes MACs also the code suggests Flops for historical reasons
        model = self._executable_model.pytorch_model
        macs_analysis = FlopCountAnalysis(model, self._sample_batch_input)
        macs_analysis.unsupported_ops_warnings(False)
        self._layer_macs = macs_analysis.by_module()
        self._q_context = self._get_q_context()
        return self

    def compute_metric_for_layer(self, layer_key: str) -> dict[str, float]:
        assert self._layer_macs is not None
        if layer_key not in self._layer_macs:
            return {}
        # the library considers batch size within the MACs results
        macs = self._layer_macs[layer_key] / self._batch_size
        a_bits, w_bits = self._get_a_w_bits(layer_key)
        return {
            "MACs": macs,
            "BOPs": macs * a_bits * w_bits
            # BOPs calculation following [van Baalen, 2020]
        }

    def _get_q_context(self):
        if self._executable_model.has_compression_context(TorchMixedQuantizationExecutor.CONTEXT_KEY):
            return self._executable_model.get_compression_context(TorchMixedQuantizationExecutor.CONTEXT_KEY)
        # if not existing create empty one -> no logic branching below
        return QuantizationContext()

    def _get_a_w_bits(self, layer_key) -> tuple[int, int]:
        if layer_key in self._q_context:
            layer_q_context = self._q_context[layer_key]
            return layer_q_context.activation_bits, layer_q_context.weight_bits
        return 32, 32

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._layer_macs = None
        self._executable_model = None
        self._q_context = None


class TorchConvLayerShapeExtractor(AMetricExtractor):

    def __init__(self):
        self._executable_model = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._executable_model = None

    def compute_metric_for_layer(self, layer_key: str) -> dict[str, float]:
        target_module = self._executable_model.module_for_key(layer_key)
        param_counts = [p.numel() for p in target_module.parameters()]
        num_param = np.sum(np.array(param_counts)).item()
        if isinstance(target_module, torch.nn.Conv2d):
            return {
                "in_channels": target_module.in_channels,
                "out_channels": target_module.out_channels,
                "kernel_size": target_module.kernel_size,
                "stride": target_module.stride,
                "params": num_param
            }
        if isinstance(target_module, torch.nn.Linear):
            return {
                "in_channels": target_module.in_features,
                "out_channels": target_module.out_features,
                "kernel_size": (1, 1),
                "stride": (0, 0),
                "params": num_param
            }
        return {}

    def __call__(self, executable_model: TorchExecutableModel):
        self._executable_model = executable_model
        return self


class PruningSparsityExtractor(AMetricExtractor):

    def __init__(self):
        self._executable_model = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._executable_model = None

    def compute_metric_for_layer(self, layer_key: str) -> dict[str, float]:
        active_policy = self._executable_model.applied_policy
        pruning_spec = self._resolve_pruning_spec(layer_key, active_policy)
        if pruning_spec:
            # not considered: disabled pruning (default should be 0.0)
            sparsity = pruning_spec.parameter_by_key("sparsity").compression_ratio
        else:
            sparsity = 0.0
        return {
            "sparsity": sparsity
        }

    @staticmethod
    def _resolve_pruning_spec(layer_key: str,
                              compression_policy: CompressionPolicy) -> LayerCompressionSpecification | None:
        if compression_policy:
            for method_key, method_spec in compression_policy.layers[layer_key].items():
                if compression_policy.group_mapping[method_key] == "prune":
                    # mutual groups, return first one found
                    return method_spec
        return None

    def __call__(self, executable_model: TorchExecutableModel):
        self._executable_model = executable_model
        return self


class TorchFeatureExtractor(AFeatureExtractor):

    def __init__(self, data_provider: ADataProvider, sensitivity_analysis: SensitivityAnalysis):
        self._data_provider = data_provider
        super(TorchFeatureExtractor, self).__init__(sensitivity_analysis)

    def _register_extractors(self) -> list[AMetricExtractor]:
        return [
            TorchMACsBOPsExtractor(self._data_provider.get_random_tensor_with_input_shape()),
            PruningSparsityExtractor(),
            TorchConvLayerShapeExtractor()
        ]
