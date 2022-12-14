import copy
import gc
from abc import ABCMeta, abstractmethod
from typing import Tuple, OrderedDict

import numpy as np

from runtime.compress.compression_policy import CompressionPolicy, CompressionProtocolEntry, \
    LayerCompressionSpecification
from runtime.model.model_handle import AModelReference, AModelFactory, AExecutableModel


class ADiscretizer(metaclass=ABCMeta):

    def discretize(self, compression_specification: LayerCompressionSpecification):
        for compression_parameter in compression_specification.compression_parameters:
            clipped_compression_ratio = np.clip(compression_parameter.compression_ratio, 0.0, 1.0)
            compression_parameter.target_discrete = self.discretize_parameter(
                clipped_compression_ratio,
                compression_parameter.reference
            )

    @abstractmethod
    def discretize_parameter(self, compression_ratio, reference) -> int:
        pass


class ACompressionExecutor(metaclass=ABCMeta):

    @abstractmethod
    def compress(self, compression_specs: dict[str, LayerCompressionSpecification],
                 executable_model: AExecutableModel) -> list[
        CompressionProtocolEntry]:
        pass

    def is_supported(self, layer_key: str) -> bool:
        # no layer specification --> not supported for action
        return self.get_layer_compression_specification(layer_key) is not None

    @abstractmethod
    def get_layer_compression_specification(self, layer_key) -> LayerCompressionSpecification:
        pass


class CompressionMethodDefinition:
    def __init__(self, method_key: str, executor: ACompressionExecutor, discretizer: ADiscretizer = None,
                 method_group: str = None):
        self.method_key = method_key
        self.executor = executor
        self.discretizer = discretizer
        self.method_group = method_group


class ACompressAdapter(metaclass=ABCMeta):

    def __init__(self, model_reference: AModelReference,
                 model_factory: AModelFactory,
                 enabled_methods: Tuple[str],
                 no_discretization=False,
                 **kwargs):
        self._compression_methods = list(filter(lambda m: m.method_key in enabled_methods,
                                                self._register_compression_methods(model_reference, model_factory,
                                                                                   **kwargs)))
        self._executors = {methods.method_key: methods.executor for methods in self._compression_methods}
        self._discretizers = {methods.method_key: methods.discretizer for methods in self._compression_methods if
                              methods.discretizer is not None}
        self._group_mapping = {method.method_key: method.method_group for method in self._compression_methods if
                               method.method_group is not None}

        self._model_reference = model_reference
        self._model_factory = model_factory
        self._no_discretization = no_discretization

    def compress(self, policy: CompressionPolicy, model: AModelReference) -> (
            list[CompressionProtocolEntry], AExecutableModel):
        protocol, compressed_model, discretized_policy = self.do_compress(policy, model)
        return protocol, compressed_model

    def do_compress(self, policy: CompressionPolicy, model_reference: AModelReference) -> (
            list[CompressionProtocolEntry], AExecutableModel, CompressionPolicy):
        executable_model = self._model_factory.to_executable_model(model_reference)
        policy.verify_mutual_groups()
        discretized_policy = copy.deepcopy(policy)
        if not self._no_discretization:
            for key, discretizer in self._discretizers.items():
                discretized_policy.discretize_for_key(key, discretizer)
        protocol = list()
        for key, executor in self._executors.items():
            compression_specs = discretized_policy.get_specs_for_key(key)
            if compression_specs:
                new_protocol_entries = executor.compress(compression_specs, executable_model)
                protocol.extend(new_protocol_entries)
        executable_model.applied_policy = discretized_policy
        executable_model.compression_protocol = protocol
        gc.collect()
        return protocol, executable_model, discretized_policy

    def get_reference_policy(self) -> CompressionPolicy:
        layer_specifications = OrderedDict.fromkeys(self._model_reference.all_layer_keys(), dict())
        for definition in self._compression_methods:
            new_layer_specifications = self._get_layer_specifications_for_method(definition)
            for layer, single_specification in new_layer_specifications.items():
                layer_specifications[layer] = {definition.method_key: single_specification} | layer_specifications[
                    layer]

        return CompressionPolicy(layer_specifications, self._group_mapping)

    def _get_layer_specifications_for_method(self, definition: CompressionMethodDefinition) -> dict[
        str, LayerCompressionSpecification]:
        all_layers = self._model_reference.all_layer_keys()
        executor = definition.executor
        layer_specifications = dict()
        for layer_key in all_layers:
            if executor.is_supported(layer_key):
                layer_specifications[layer_key] = executor.get_layer_compression_specification(layer_key)
        return layer_specifications

    @abstractmethod
    def _register_compression_methods(self, original_model: AModelReference, model_factory: AModelFactory, **kwargs) -> \
            list[
                CompressionMethodDefinition]:
        pass
