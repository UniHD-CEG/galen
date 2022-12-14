from typing import Tuple

import torch
import torch_pruning

from runtime.compress.compress_adapters import ACompressAdapter, CompressionMethodDefinition
from runtime.compress.discretizers import RoundToDiscretizer, InverseRatioDiscretizer
from runtime.compress.torch_compress.torch_executors import TorchDepPruningExecutor, TorchFp32Executor, \
    TorchInt8QuantizationExecutor, TorchMixedQuantizationExecutor
from runtime.model.torch_model import TorchModelReference, TorchModelFactory


class TorchCompressAdapter(ACompressAdapter):

    def __init__(self, model_reference: TorchModelReference, model_factory: TorchModelFactory,
                 enabled_methods: Tuple[str], no_discretization=False, channel_round_to=32, mixed_reference_bits=8,
                 x86_mixed_check=False):
        super(TorchCompressAdapter, self).__init__(model_reference, model_factory, enabled_methods,
                                                   no_discretization=no_discretization,
                                                   channel_round_to=channel_round_to,
                                                   mixed_reference_bits=mixed_reference_bits,
                                                   x86_mixed_check=x86_mixed_check)

    def _register_compression_methods(self, model_reference: TorchModelReference, model_factory: TorchModelFactory,
                                      **kwargs) -> \
            list[CompressionMethodDefinition]:
        channel_round_to = kwargs.pop("channel_round_to", 1)
        mixed_reference_bits = kwargs.pop("mixed_reference_bits", 8)
        x86_mixed_check = kwargs.pop("x86_mixed_check", False)
        return [
            CompressionMethodDefinition(
                method_key="p-conv",
                executor=TorchDepPruningExecutor(model_reference, model_factory, "p-conv"),
                discretizer=RoundToDiscretizer(channel_round_to),
                method_group="prune"
            ),
            CompressionMethodDefinition(
                method_key="p-lin",
                executor=TorchDepPruningExecutor(model_reference,
                                                 model_factory,
                                                 "p-lin",
                                                 prune_method=torch_pruning.prune_linear_out_channel,
                                                 sub_desc="lin-out-L1",
                                                 target_layer_type=torch.nn.Linear),
                discretizer=RoundToDiscretizer(channel_round_to),
                method_group="prune"
            ),
            CompressionMethodDefinition(
                method_key="q-fp32",
                executor=TorchFp32Executor(model_reference, "q-fp32"),
                method_group="quantize"
            ),
            CompressionMethodDefinition(
                method_key="q-int8",
                executor=TorchInt8QuantizationExecutor(model_reference, "q-int8"),
                method_group="quantize"
            ),
            CompressionMethodDefinition(
                method_key="q-mixed",
                executor=TorchMixedQuantizationExecutor(model_reference, "q-mixed",
                                                        reference_bits=mixed_reference_bits,
                                                        x86_mixed_check=x86_mixed_check),
                discretizer=InverseRatioDiscretizer(),
                method_group="quantize"
            )
        ]
