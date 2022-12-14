import copy
from abc import ABCMeta, abstractmethod
from typing import Tuple

import torch

from runtime.compress.compression_policy import CompressionPolicy, CompressionProtocolEntry
from runtime.model.model_handle import ModelPhase, AExecutableModel, AModelReference, AModelFactory


class CompressionContext(metaclass=ABCMeta):

    @abstractmethod
    def set_model_phase(self, phase: ModelPhase):
        pass

    @abstractmethod
    def remove_compression(self):
        pass


class TorchModelReference(AModelReference):
    def __init__(self,
                 reference_model: torch.nn.Module,
                 batch_input_shape: Tuple[int, ...],
                 frozen_layers: dict[str, list[str]]):
        self._batch_input_shape = batch_input_shape
        self._reference_model = reference_model
        self._frozen_layers = frozen_layers
        self._layer_dict = {layer_key: module for layer_key, module in self._reference_model.named_modules() if
                            not [*module.children()]}

    def all_layer_keys(self) -> list[str]:
        return [*self._layer_dict]

    def all_layer_keys_for_type(self, module_type: type):
        return [layer_key for layer_key, module in self._layer_dict.items() if isinstance(module, module_type)]

    def weight_shape_of_layer_by_key(self, layer_key):
        return self._layer_dict[layer_key].weight.shape

    @property
    def batch_input_shape(self) -> Tuple[int, ...]:
        return self._batch_input_shape

    def frozen_layers(self, method_key: str) -> list[str]:
        return self._frozen_layers.get(method_key, list())


class TorchExecutableModel(TorchModelReference, AExecutableModel):

    def __init__(self, pytorch_model: torch.nn.Module, batch_input_shape: Tuple[int, ...], target_device: torch.device,
                 frozen_layers: dict[str, list[str, ...]]):
        super(TorchExecutableModel, self).__init__(pytorch_model, batch_input_shape, frozen_layers)
        # all layer_keys without any children
        self._method_state: dict[str, CompressionContext] = dict()
        self._applied_policy = None
        self._target_device = target_device
        self._compression_protocol = None

    def module_for_key(self, layer_key) -> torch.nn.Module:
        return self._layer_dict[layer_key]

    @property
    def pytorch_model(self):
        return self._reference_model

    @property
    def target_device(self) -> torch.device:
        return self._target_device

    @property
    def applied_policy(self) -> CompressionPolicy:
        return self._applied_policy

    @property
    def compression_protocol(self) -> list[CompressionProtocolEntry]:
        return self._compression_protocol

    @applied_policy.setter
    def applied_policy(self, policy):
        self._applied_policy = policy

    @compression_protocol.setter
    def compression_protocol(self, protocol):
        self._compression_protocol = protocol

    def remove_all_contexts(self):
        for context in self._method_state.values():
            context.remove_compression()
        self._method_state = dict()

    def register_compression_context(self, context_identifier: str, context: CompressionContext):
        if context_identifier in self._method_state:
            raise Exception(f"Context with identifier {context_identifier} already present for model")
        self._method_state[context_identifier] = context

    def remove_compression_context(self, context_identifier: str):
        if context_identifier not in self._method_state:
            raise Exception(f"Context with identifier {context_identifier} not present for model")
        self._method_state.pop(context_identifier)

    def has_compression_context(self, context_identifier: str):
        return context_identifier in self._method_state

    def get_compression_context(self, context_identifier: str):
        if context_identifier not in self._method_state:
            raise Exception(f"Context with identifier {context_identifier} not present for model")
        return self._method_state.get(context_identifier)


class TorchModelFactory(AModelFactory):

    def __init__(self,
                 torch_model: torch.nn.Module,
                 batch_input_shape: Tuple[int, ...],
                 target_device: torch.device,
                 frozen_layers: dict[str, list[str]]):
        self._torch_reference_model = torch_model
        self._target_device = target_device
        self._reference_model = TorchModelReference(self._torch_reference_model, batch_input_shape, frozen_layers)

    def get_reference_model(self):
        return self._reference_model

    def to_executable_model(self, model_reference: TorchModelReference) -> TorchExecutableModel:
        new_model = copy.deepcopy(self._torch_reference_model)
        new_model.to(self._target_device)
        return TorchExecutableModel(new_model,
                                    model_reference.batch_input_shape,
                                    self._target_device,
                                    model_reference._frozen_layers)
