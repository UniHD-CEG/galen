import copy
import pickle
from typing import Tuple


class NoCompressionRatioSetException(Exception):
    def __init__(self):
        super(NoCompressionRatioSetException, self) \
            .__init__(f"No compression ratio parameter set for a layer")


class NotDiscretizedException(Exception):
    def __init__(self):
        super(NotDiscretizedException, self) \
            .__init__(f"LayerCompressionSpecification for the layer is not discretized before execution.")


class CompressionRatioParameter:
    def __init__(self, parameter_key: str, reference: int, compression_ratio, target_discrete, bounds=(0.0, 1.0)):
        self._parameter_key = parameter_key
        self._reference = reference
        self._compression_ratio = compression_ratio
        self._target_discrete = target_discrete
        self._bounds = bounds

    @classmethod
    def with_reference(cls, reference: int, parameter_key: str, initial_ratio=0.0,
                       bounds=(0.0, 1.0)) -> 'CompressionRatioParameter':
        return cls(parameter_key, reference, initial_ratio, None, bounds)

    @property
    def parameter_key(self):
        return self._parameter_key

    @property
    def bounds(self):
        return self._bounds

    @property
    def reference(self) -> int:
        return self._reference

    @property
    def compression_ratio(self) -> float:
        return self._compression_ratio

    @compression_ratio.setter
    def compression_ratio(self, value):
        self._compression_ratio = value
        self._target_discrete = None

    @property
    def target_discrete(self) -> int:
        self._verify_discretized()
        return self._target_discrete

    @target_discrete.setter
    def target_discrete(self, discrete_value):
        self._target_discrete = discrete_value

    def _verify_discretized(self):
        if self._target_discrete is None:
            raise NotDiscretizedException()


class LayerCompressionSpecification:
    def __init__(self, layer_key, is_active: bool, compression_parameters: Tuple[CompressionRatioParameter]):
        self._layer_key = layer_key
        self._is_active = is_active
        self._compression_parameters = compression_parameters

    @classmethod
    def create_initial(cls, layer_key: str, compression_parameters: Tuple[CompressionRatioParameter] = tuple(
        []), default_active_state=False) -> 'LayerCompressionSpecification':
        return cls(layer_key, default_active_state, compression_parameters)

    @property
    def layer_key(self) -> str:
        return self._layer_key

    @property
    def is_active(self) -> bool:
        return self._is_active

    @property
    def compression_parameters(self) -> Tuple[CompressionRatioParameter]:
        return self._compression_parameters

    def has_parameters(self) -> bool:
        return len(self._compression_parameters) != 0

    def replace_parameters(self, parameters: Tuple[CompressionRatioParameter]):
        self._compression_parameters = parameters

    def replace_single_parameter(self, new_parameter: CompressionRatioParameter):
        self._compression_parameters = tuple([new_parameter if p.parameter_key == new_parameter.parameter_key else p
                                              for p in self._compression_parameters])

    def parameter_by_key(self, parameter_key) -> CompressionRatioParameter:
        for compression_parameter in self._compression_parameters:
            if compression_parameter.parameter_key == parameter_key:
                return compression_parameter
        raise Exception(f"Parameter with name {parameter_key} in specification for layer {self._layer_key} not found")

    @is_active.setter
    def is_active(self, is_active: bool):
        self._is_active = is_active


class CompressionPolicy:
    def __init__(self, layers: dict[str, dict[str, LayerCompressionSpecification]], group_mapping: dict[str, str]):
        self._layers = layers
        self._group_mappings = group_mapping

    @property
    def layers(self) -> dict[str, dict[str, LayerCompressionSpecification]]:
        return self._layers

    @property
    def group_mapping(self):
        return self._group_mappings

    def get_included_methods(self):
        return self._group_mappings.keys()

    def merge_policies(self, other_policy: 'CompressionPolicy'):
        self._group_mappings.update(other_policy._group_mappings)
        for layer_key, layer_dict in self._layers.items():
            layer_dict.update(other_policy._layers[layer_key])
            for method_key, method_spec in other_policy._layers[layer_key].items():
                if method_spec.is_active:
                    self.activate_compression(layer_key, method_key)

    def verify_mutual_groups(self):
        for layer_key, layer_specifications in self.layers.items():
            found_groups = set()
            for method_key in layer_specifications.keys():
                if method_key in self._group_mappings and layer_specifications[method_key].is_active:
                    group = self._group_mappings[method_key]
                    if group in found_groups:
                        raise Exception(
                            f"Mutual group violation: found compression methods for layer {layer_key}: {layer_specifications.keys()}")
                    found_groups.add(group)

    def discretize_for_key(self, key: str, discretizer):
        for layer, c_dict in self._layers.items():
            if key in c_dict and c_dict[key].is_active:
                discretizer.discretize(c_dict[key])

    def get_specs_for_key(self, key) -> dict[str, LayerCompressionSpecification]:
        return {layer_key: spec for layer_key, compression_dict in self._layers.items() for comp_key, spec in
                compression_dict.items() if comp_key == key and spec.is_active}

    def activate_compression(self, layer_key: str, method_key: str):
        self._layers[layer_key][method_key].is_active = True
        for other_key, other_spec in self._layers[layer_key].items():
            if other_key != method_key and self._group_mappings[other_key] == self._group_mappings[method_key]:
                other_spec.is_active = False

    def get_clean_policy(self):
        duplicate = copy.deepcopy(self)
        for layer_methods in duplicate._layers.values():
            for method_key, method in list(layer_methods.items()):
                if not method.is_active:
                    layer_methods.pop(method_key)
        return duplicate


class CompressionProtocolEntry:
    def __init__(self, before, layer_key, result, compression_type, compression_params: Tuple):
        self._before = before
        self._layer_key = layer_key
        self._result = result
        self._compression_type = compression_type
        self._compression_params = compression_params

    @property
    def layer_key(self):
        return self._layer_key

    @property
    def compression_type(self):
        return self._compression_type

    @property
    def before(self):
        return self._before

    @property
    def result(self):
        return self._result

    @property
    def compression_params(self) -> Tuple:
        return self._compression_params


def load_policy(policy_path) -> CompressionPolicy:
    with open(policy_path, 'rb') as file:
        data = pickle.load(file)
        if isinstance(data, CompressionPolicy):
            return data
        if isinstance(data, dict) and "_policy" in data:
            return data["_policy"]
    raise Exception("Object is not a policy / does not contain a policy")
