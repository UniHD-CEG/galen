import copy

import numpy as np

from runtime.agent.agent import TargetLayer, ADdpgAgent
from runtime.compress.compression_policy import CompressionPolicy
from runtime.controller.algorithmic_mode import AlgorithmicMode
from runtime.feature_extraction.feature_extractor import ModelFeatures, LayerFeatures


class IterativeSingleLayerPruningAgent(ADdpgAgent):
    """
    - layer-by-layer prediction of pruning sparsity
    - using features specific to current target layer only
    - expects dependent episodes / delta sensitivity

    """

    def supported_method(self) -> list[str]:
        return ["p-conv", "p-lin"]

    STATE_SHAPE = (11,)

    def _get_state_shape(self):
        return self.STATE_SHAPE

    def _get_num_actions(self):
        return 1

    def _extract_layer_state(self, target_layer: TargetLayer, features: ModelFeatures) -> np.ndarray:
        if target_layer.key not in features.features:
            raise Exception(f"No features for target layer {target_layer.key}!")
        layer_features = features.features[target_layer.key]
        self._assert_keys(target_layer, layer_features)

        net_macs = self._acc_macs(0, len(features.features), features)

        """
            state features, shape=(11,):
            - [ 0] layer idx                                 (x)
            - [ 1] MACs layer                                (x)
            - [ 2] n: number of output channels              (x)
            - [ 3] c: number of input channels               (x)
            - [ 4] k: kernel size                            (x)
            - [ 5] s: stride                                 (x)
            - [ 6] relative acc. MACs before layer           (x)
            - [ 7] relative acc. MACs following the layer    (x)
            - [ 8] Sens. inc. pruning sparsity               (x)
            - [ 9] Sens. dec. pruning sparsity               (x)
            - [10] Current Sparsity                          (x)
            
            (*) = non-normalized features
            (x) = standardized features 
            (/) = min-max normalization
        """

        state = np.zeros(self.STATE_SHAPE)
        state[0] = target_layer.step
        state[1] = layer_features.metrics["MACs"]
        state[2] = layer_features.metrics["out_channels"]
        state[3] = layer_features.metrics["in_channels"]
        state[4] = layer_features.metrics["kernel_size"][0]
        state[5] = layer_features.metrics["stride"][0]
        state[6] = self._acc_macs(0, target_layer.step, features) / net_macs
        state[7] = self._acc_macs(target_layer.step + 1, len(features.features), features) / net_macs
        state[8] = layer_features.sensitivity.policy_results[f"{target_layer.method_key}-sparsity-inc"]
        state[9] = layer_features.sensitivity.policy_results[f"{target_layer.method_key}-sparsity-dec"]
        state[10] = layer_features.metrics["sparsity"]

        state = state.reshape((1, -1))
        # self._min_max_scaler.partial_fit(state[:, :6])
        self._standard_scaler.partial_fit(state)
        # min/max for basic features
        # state[:, :6] = self._min_max_scaler.transform(state[:, :6])
        # centralization & standardization for sensitivity -> gaussian distribution
        state = self._standard_scaler.transform(state)
        # accumulated mac counts and sparsity is already normalized
        return state.reshape((-1,))

    def _create_policy_with_prediction(self, target_layer: TargetLayer, predicted_actions: np.ndarray,
                                       reference_policy: CompressionPolicy) -> CompressionPolicy:
        duplicated_reference = copy.deepcopy(reference_policy)
        duplicated_reference.activate_compression(target_layer.key, target_layer.method_key)
        compression_parameter = duplicated_reference.layers[target_layer.key][
            target_layer.method_key].parameter_by_key("sparsity")
        compression_parameter.compression_ratio = predicted_actions.item()
        return duplicated_reference

    @staticmethod
    def _assert_keys(target_layer: TargetLayer, layer_features: LayerFeatures):
        if {"MACs", "in_channels", "out_channels", "kernel_size", "stride", "sparsity"} > layer_features.metrics.keys():
            raise Exception(f"Layer {target_layer.key} misses features, found: {layer_features.metrics.keys()}")
        if {"sparsity-inc", "sparsity-dec"} > layer_features.sensitivity.policy_results.keys():
            raise Exception(
                f"Layer {target_layer.key} misses sensitivity results, found: {layer_features.metrics.keys()}")

    def get_search_id(self):
        return f"p_it_{self._reward_function.get_search_id()}"


class IndependentSingleLayerPruningAgent(IterativeSingleLayerPruningAgent):
    STATE_SHAPE = (19,)

    def _get_state_shape(self):
        return self.STATE_SHAPE

    def alg_mode(self) -> AlgorithmicMode:
        return AlgorithmicMode.INDEPENDENT

    def _extract_layer_state(self, target_layer: TargetLayer, features: ModelFeatures) -> np.ndarray:
        layer_features = features.features[target_layer.key]
        pruning_method_key = self._get_pruning_method(layer_features)
        net_macs = self._acc_macs(0, len(features.features), features)

        state = np.zeros(self.STATE_SHAPE)
        state[0] = target_layer.step
        state[1] = layer_features.metrics["MACs"]
        state[2] = layer_features.metrics["out_channels"]
        state[3] = layer_features.metrics["in_channels"]
        state[4] = layer_features.metrics["kernel_size"][0]
        state[5] = layer_features.metrics["stride"][0]
        state[6] = layer_features.metrics["params"]
        state[6] = self._acc_macs(0, target_layer.step, features) / net_macs
        state[7] = self._acc_macs(target_layer.step + 1, len(features.features), features) / net_macs
        state[8:18] = self._get_sensitivity_slice(pruning_method_key, 10, layer_features)
        state[18] = layer_features.metrics["sparsity"]

        state = state.reshape((1, -1))
        self._standard_scaler.partial_fit(state)
        state = self._standard_scaler.transform(state)
        return state.reshape((-1,))

    def _get_pruning_method(self, layer_features: LayerFeatures):
        if any(method.startswith("p-lin") for method in layer_features.sensitivity.policy_results.keys()):
            return "p-lin-sparsity"
        return "p-conv-sparsity"

    def get_search_id(self):
        return f"p_ind_{self._reward_function.get_search_id()}"
