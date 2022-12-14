import copy
from abc import ABCMeta
from typing import Tuple

import torch.nn
import torch.utils.hooks
import torch_pruning as tp

from runtime.compress.compress_adapters import ACompressionExecutor
from runtime.compress.compression_policy import CompressionProtocolEntry, CompressionRatioParameter, \
    LayerCompressionSpecification
from runtime.compress.torch_compress.fake_asymetric_linear_quantization import FakeQuantizeOp
from runtime.compress.torch_compress.model_info import ModelInfo
from runtime.model.model_handle import ModelPhase
from runtime.model.torch_model import TorchExecutableModel, CompressionContext, TorchModelReference, TorchModelFactory


class ATorchCompressionExecutor(ACompressionExecutor, metaclass=ABCMeta):
    def __init__(self, method_key):
        self._method_key = method_key

    @property
    def method_key(self):
        return self._method_key


class TorchDepPruningExecutor(ATorchCompressionExecutor):

    def __init__(self, model_reference: TorchModelReference,
                 model_factory: TorchModelFactory,
                 method_key: str,
                 strategy=tp.strategy.L1Strategy,
                 prune_method=tp.prune_conv_out_channel,
                 sub_desc="conv-channel-L1",
                 target_layer_type=torch.nn.Conv2d):
        super(TorchDepPruningExecutor, self).__init__(method_key)
        self._model_reference = model_reference
        self._model_factory = model_factory
        self._strategy = strategy()
        self._prune_method = prune_method
        self._sub_desc = sub_desc
        self._target_layer_type = target_layer_type
        self._independent_prunable = self._resolve_independent_prunable_layers()

    def compress(self, compression_specs: dict[str, LayerCompressionSpecification],
                 model_handle: TorchExecutableModel) -> \
            list[CompressionProtocolEntry]:
        random_input = torch.randn(model_handle.batch_input_shape).to(model_handle.target_device)
        dep_graph = tp.DependencyGraph().build_dependency(model_handle.pytorch_model, example_inputs=random_input)

        def _prune(l_key: str, c_spec: CompressionRatioParameter):
            target_module = model_handle.module_for_key(l_key)
            amount_to_remove = c_spec.reference - c_spec.target_discrete
            prune_idxs = self._strategy(target_module.weight, amount_to_remove)
            plan = dep_graph.get_pruning_plan(target_module, self._prune_method, prune_idxs)
            before = self._fetch_shapes(plan)
            plan.exec()
            after = self._fetch_shapes(plan)
            return before, after

        protocol = list()
        for layer_key, spec in compression_specs.items():
            if spec.is_active:
                self._verify_layer(layer_key)
                compression_parameter = spec.compression_parameters[0]
                before_shapes, after_shapes = _prune(layer_key, compression_parameter)
                protocol.extend(self._assemble_protocol(before_shapes, after_shapes, spec))

        return protocol

    def _verify_layer(self, layer_key):
        if layer_key not in self._independent_prunable:
            raise Exception(f"Layer {layer_key} not supported for pruning by dependent pruning executor!")

    def get_layer_compression_specification(self, layer_key) -> LayerCompressionSpecification:
        if layer_key in self._independent_prunable:
            weight_shape = self._model_reference.weight_shape_of_layer_by_key(layer_key)
            # for out channels (conv channel pruning) and out features
            compression_parameter = CompressionRatioParameter.with_reference(weight_shape[0], parameter_key="sparsity")
            return LayerCompressionSpecification.create_initial(layer_key, tuple([compression_parameter]))
        return None

    def _resolve_independent_prunable_layers(self) -> set[str]:
        prunable_keys = self._model_reference.all_layer_keys_for_type(self._target_layer_type)
        filtered_keys = list(
            filter(lambda l_key: l_key not in self._model_reference.frozen_layers(self.method_key), prunable_keys))
        return self._remove_dependent_layers(filtered_keys)

    def _remove_dependent_layers(self, prunable_keys: list[str]) -> set[str]:
        executable_model = self._model_factory.to_executable_model(self._model_reference)
        dep_graph = tp.DependencyGraph()
        random_input = torch.randn(self._model_reference.batch_input_shape).to(executable_model.target_device)
        dep_graph.build_dependency(executable_model.pytorch_model,
                                   example_inputs=random_input)

        dependent_keys = set()
        for layer_key in prunable_keys:
            if layer_key not in dependent_keys:
                new_dependent_keys = self._resolve_dependent_keys(dep_graph, layer_key, executable_model)
                dependent_keys.update(new_dependent_keys)
        return set(prunable_keys).difference(dependent_keys)

    def _resolve_dependent_keys(self, dep_graph: tp.DependencyGraph, layer_key: str, model: TorchExecutableModel) -> \
            list[str]:
        current_module = model.module_for_key(layer_key)
        # dummy sparsity to resolve dependencies only
        dummy_idxs = self._strategy(current_module.weight, amount=0.5)
        pruning_plan = dep_graph.get_pruning_plan(current_module, self._prune_method, idxs=dummy_idxs)
        if not dep_graph.check_pruning_plan(pruning_plan):
            raise Exception("Zero pruning for at least one channel")
        return self._extract_dependent_layer_names(pruning_plan)

    def _extract_dependent_layer_names(self, pruning_plan) -> list[str]:
        dependent_keys = list()
        for dep_tuple in pruning_plan.plan[1:]:  # skip first -> self dependency
            dependency = dep_tuple[0]
            # same handler type means this is a blocking dependency
            if isinstance(dependency.handler, type(self._prune_method)):
                dependent_keys.append(self.get_module_key(dependency))
        return dependent_keys

    @staticmethod
    def get_module_key(dependency):
        return dependency.target.name.split(" ")[0]

    def _fetch_shapes(self, plan: tp.PruningPlan):
        shapes = dict()
        for dep_tuple in plan.plan:
            dependency = dep_tuple[0]
            target_module = dependency.target.module
            if hasattr(target_module, 'weight'):
                # no weight - no savings -> no need to track
                shapes[self.get_module_key(dependency)] = target_module.weight.detach().shape
        return shapes

    @staticmethod
    def _assemble_protocol(before_shapes: dict[str, Tuple[int, ...]], after_shapes: dict[str, Tuple[int, ...]],
                           spec: LayerCompressionSpecification):
        protocol = list()
        for layer_key, before_shape in before_shapes.items():
            compression_params = spec.compression_parameters if layer_key == spec.layer_key else None
            next_entry = CompressionProtocolEntry(
                layer_key=layer_key,
                before=before_shape,
                result=after_shapes[layer_key],
                compression_type='dep-prune',
                compression_params=compression_params
            )
            protocol.append(next_entry)
        return protocol


class TorchMixedQuantizationExecutor(ATorchCompressionExecutor):
    CONTEXT_KEY = "quantization_context"
    ACTIVATION_KEY = 'activation'
    WEIGHT_KEY = 'weight'

    def __init__(self, model_reference: TorchModelReference,
                 method_key: str,
                 supported_layer_types: Tuple[type, ...] = tuple([torch.nn.Conv2d, torch.nn.Linear]),
                 mode: str = 'mixed',
                 reference_bits: int = 8,
                 x86_mixed_check: bool = False):
        super(TorchMixedQuantizationExecutor, self).__init__(method_key)
        self._mode = mode
        self._reference_bits = reference_bits
        self._x86_mixed_check = x86_mixed_check
        self._compressible_layers = self._extract_compressible_layers(model_reference, supported_layer_types)

    def compress(self, compression_specs: dict[str, LayerCompressionSpecification],
                 executable_model: TorchExecutableModel) -> list[CompressionProtocolEntry]:
        quantization_context = self.get_or_create_q_context(executable_model)
        protocol = list()
        for layer_key, compression_spec in compression_specs.items():
            if layer_key in quantization_context:
                old_q_context = quantization_context.pop(layer_key)
                old_q_context.remove_compression()
            q_layer_context = QuantizationLayerContext(compression_spec)
            quantization_context[layer_key] = q_layer_context

            target_layer = executable_model.module_for_key(layer_key)
            pre_hook = target_layer.register_forward_pre_hook(q_layer_context.pre_forward_hook)
            q_layer_context.set_pre_hook_handle(pre_hook)
            post_hook = target_layer.register_forward_hook(q_layer_context.post_forward_hook)
            q_layer_context.set_post_hook_handle(post_hook)
            protocol.append(self._assemble_protocol(compression_spec))
        return protocol

    def get_layer_compression_specification(self, layer_key) -> LayerCompressionSpecification:
        if self.is_quantizable(layer_key):
            weight_ratio = CompressionRatioParameter.with_reference(reference=self._reference_bits,
                                                                    parameter_key=self.WEIGHT_KEY)
            activation_ratio = CompressionRatioParameter.with_reference(reference=self._reference_bits,
                                                                        parameter_key=self.ACTIVATION_KEY)
            return LayerCompressionSpecification.create_initial(layer_key, tuple([weight_ratio, activation_ratio]))

        return None

    def is_quantizable(self, layer_key):
        return layer_key in self._compressible_layers

    def _extract_compressible_layers(self, model_reference: TorchModelReference,
                                     supported_layer_types: Tuple[type, ...]) -> set[str]:
        all_supported = set()
        for supported_type in supported_layer_types:
            unfiltered = model_reference.all_layer_keys_for_type(supported_type)
            model_info = ModelInfo(model_reference)
            no_frozen = [x for x in unfiltered if x not in model_reference.frozen_layers(self.method_key)]
            no_incompatible = [x for x in no_frozen if self._is_layer_compatible(x, model_info)]
            all_supported.update(no_incompatible)
        return all_supported

    def _is_layer_compatible(self, layer_key, model_info) -> bool:
        layer_info = model_info.get_info_for_layer(layer_key)
        if self._x86_mixed_check:
            if isinstance(layer_info.module, torch.nn.Conv2d):
                module = layer_info.module
                if module.in_channels % 16 != 0:
                    return False
                if module.out_channels % 16 != 0:
                    return False
            if isinstance(layer_info.module, torch.nn.Linear):
                module = layer_info.module
                if module.out_features % 16 != 0:
                    return False
            return True
        if isinstance(layer_info.module, torch.nn.Conv2d):
            min_output_size = min(layer_info.output_size[2], layer_info.output_size[3])
            if min_output_size < 2:
                return False
            module = layer_info.module
            if module.in_channels % 32 != 0:
                return False
            if module.out_channels % 8 != 0:
                return False
            if module.groups != 1:
                return False
        if isinstance(layer_info.module, torch.nn.Linear):
            module = layer_info.module
            if module.out_features % 8 != 0:
                return False

        return True

    def _assemble_protocol(self, compression_spec: LayerCompressionSpecification):
        protocol = CompressionProtocolEntry(
            layer_key=compression_spec.layer_key,
            before=None,
            result={
                'w_bit': compression_spec.parameter_by_key(self.WEIGHT_KEY).target_discrete,
                'a_bit': compression_spec.parameter_by_key(self.ACTIVATION_KEY).target_discrete
            },
            compression_type=f"quantize-{self._mode}",
            compression_params=compression_spec.compression_parameters
        )
        return protocol

    def get_or_create_q_context(self, model: TorchExecutableModel):
        if not model.has_compression_context(self.CONTEXT_KEY):
            model.register_compression_context(self.CONTEXT_KEY, QuantizationContext())
        return model.get_compression_context(self.CONTEXT_KEY)


class TorchInt8QuantizationExecutor(TorchMixedQuantizationExecutor):

    def __init__(self, model_reference: TorchModelReference, method_key: str,
                 supported_layer_types: Tuple[type, ...] = tuple([torch.nn.Conv2d, torch.nn.Linear])):
        super(TorchInt8QuantizationExecutor, self).__init__(model_reference, method_key, supported_layer_types,
                                                            mode="int8")

    def compress(self, compression_specs: dict[str, LayerCompressionSpecification],
                 executable_model: TorchExecutableModel) -> \
            list[
                CompressionProtocolEntry]:
        adapted_specs = copy.deepcopy(compression_specs)
        for spec in adapted_specs.values():
            weight_ratio = CompressionRatioParameter(parameter_key=self.WEIGHT_KEY, reference=8, compression_ratio=0.0,
                                                     target_discrete=8)
            activation_ratio = CompressionRatioParameter(parameter_key=self.ACTIVATION_KEY, reference=8,
                                                         compression_ratio=0.0, target_discrete=8)
            spec.replace_parameters(tuple([weight_ratio, activation_ratio]))
        return super(TorchInt8QuantizationExecutor, self).compress(adapted_specs, executable_model)

    def get_layer_compression_specification(self, layer_key) -> LayerCompressionSpecification:
        if self.is_quantizable(layer_key):
            return LayerCompressionSpecification.create_initial(layer_key)
        return None

    def _is_layer_compatible(self, layer_key, model_info) -> bool:
        return True


class TorchFp32Executor(TorchMixedQuantizationExecutor):

    def __init__(self, model_reference: TorchModelReference, method_key: str,
                 supported_layer_types: Tuple[type, ...] = tuple([torch.nn.Conv2d, torch.nn.Linear])):
        super(TorchFp32Executor, self).__init__(model_reference, method_key, supported_layer_types, mode="fp32")

    def compress(self, compression_specs: dict[str, LayerCompressionSpecification],
                 executable_model: TorchExecutableModel) -> \
            list[
                CompressionProtocolEntry]:
        quantization_contex = self.get_or_create_q_context(executable_model)
        for layer_key in compression_specs.keys():
            # is active was checked by compress adapter
            if layer_key in quantization_contex:
                quantization_contex.pop(layer_key)
        return list()

    def get_layer_compression_specification(self, layer_key) -> LayerCompressionSpecification:
        if self.is_quantizable(layer_key):
            return LayerCompressionSpecification.create_initial(layer_key, default_active_state=True)
        return None

    def _is_layer_compatible(self, layer_key, model_info) -> bool:
        return True


class QuantizationLayerContext:

    def __init__(self, compression_spec):
        self._compression_spec = compression_spec
        self._original_weight = None
        self._pre_hook_handle = None
        self._post_hook_handle = None
        self._phase = ModelPhase.VALIDATION

    @property
    def weight_bits(self):
        return self._compression_spec.parameter_by_key("weight").target_discrete

    @property
    def activation_bits(self):
        return self._compression_spec.parameter_by_key("activation").target_discrete

    def set_model_phase(self, phase: ModelPhase):
        self._phase = phase

    def pre_forward_hook(self, module: torch.nn.Module, input: torch.Tensor):
        self._original_weight = module.weight
        quantized_weight = FakeQuantizeOp.apply(self._original_weight, self.weight_bits)
        module.weight = torch.nn.Parameter(quantized_weight)

        quantized_input = FakeQuantizeOp.apply(input[0], self.activation_bits)

        return quantized_input

    def set_pre_hook_handle(self, handle: torch.utils.hooks.RemovableHandle):
        self._pre_hook_handle = handle

    def post_forward_hook(self, module, input, output):
        module.weight = self._original_weight

    def set_post_hook_handle(self, handle: torch.utils.hooks.RemovableHandle):
        self._post_hook_handle = handle

    def remove_hooks(self):
        if self._pre_hook_handle is not None:
            self._pre_hook_handle.remove()
        if self._post_hook_handle is not None:
            self._post_hook_handle.remove()


class QuantizationContext(CompressionContext):

    def __init__(self):
        self._layer_contexts: dict[str, QuantizationLayerContext] = dict()

    def set_model_phase(self, phase: ModelPhase):
        for layer_context in self._layer_contexts.values():
            layer_context.set_model_phase(phase)

    def __getitem__(self, item_key: str) -> QuantizationLayerContext:
        return self._layer_contexts[item_key]

    def __setitem__(self, key: str, value: QuantizationLayerContext):
        self._layer_contexts[key] = value

    def __contains__(self, key) -> bool:
        return key in self._layer_contexts

    def pop(self, key) -> QuantizationLayerContext:
        return self._layer_contexts.pop(key)

    def remove_compression(self):
        for context in self._layer_contexts.values():
            context.remove_hooks()
        self._layer_contexts = dict()
