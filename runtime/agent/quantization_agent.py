import copy
from dataclasses import dataclass

import numpy as np
import torch

from runtime.agent.agent import ADdpgAgent, GenericAgentConfig
from runtime.compress.compression_policy import CompressionPolicy, LayerCompressionSpecification
from runtime.controller.algorithmic_mode import AlgorithmicMode


@dataclass
class LastAction:
    layer_key: str
    weight: float = 0.0
    activation: float = 0.0
    threshold_ratio: float = 0.0


class QuantizationAgentConfig(GenericAgentConfig):

    def __init__(self, **kwargs):
        super(QuantizationAgentConfig, self).__init__(**kwargs)
        self.int8_threshold = float(kwargs.pop("q_agent_int8_threshold", 0.2))
        self.mixed_threshold = float(kwargs.pop("q_agent_mixed_threshold", 0.5))


class IterativeSingleLayerQuantizationAgent(ADdpgAgent):
    STATE_SHAPE = (16,)

    def __init__(self, target_device: torch.device, agent_config=QuantizationAgentConfig()):
        super().__init__(target_device, agent_config=agent_config)
        self._cfg = agent_config
        self._last_ratios = dict()

    def supported_method(self) -> list[str]:
        return ["q-mixed", "q-fp32", "q-int8"]

    def _get_state_shape(self):
        return self.STATE_SHAPE

    def _get_num_actions(self):
        return 2

    def _extract_layer_state(self, target_layer, features):
        layer_features = features.features[target_layer.key]

        net_bops = self._acc_bops(0, len(features.features), features)
        last_action = self._last_ratios.pop(target_layer.key, LastAction(target_layer.key))

        """
            state features, shape=(11,):
            - [ 0] layer idx                                 (x)
            - [ 1] BOPs layer                                (x)
            - [ 2] n: number of output channels              (x)
            - [ 3] c: number of input channels               (x)
            - [ 4] k: kernel size                            (x)
            - [ 5] s: stride                                 (x)
            - [ 6] n: number of parameters                   (x)
            - [ 6] relative acc. BOPs before layer           (x)
            - [ 7] relative acc. BOPs following the layer    (x)
            - [ 8] Sens. inc. mixed activation ratio         (x)
            - [ 9] Sens. dec. mixed activation ratio         (x)
            - [10] Sens. inc. mixed weight ratio             (x)
            - [11] Sens. dec. mixed weight ratio             (x)
            - [12] Sens. fp32                                (x)
            - [13] Sens. int8                                (x)
            - [14] Current activation ratio                  (x)
            - [15] Current weight ratio                      (x)

            (*) = non-normalized features
            (x) = standardized features
            (/) = min-max normalization
        """

        state = np.zeros(self.STATE_SHAPE)
        state[0] = target_layer.step
        state[1] = layer_features.metrics["BOPs"]
        state[2] = layer_features.metrics["out_channels"]
        state[3] = layer_features.metrics["in_channels"]
        state[4] = layer_features.metrics["kernel_size"][0]
        state[5] = layer_features.metrics["stride"][0]
        state[6] = layer_features.metrics["params"]
        state[6] = self._acc_bops(0, target_layer.step, features) / net_bops
        state[7] = self._acc_bops(target_layer.step + 1, len(features.features), features) / net_bops
        state[8] = self._get_sensitivity("q-mixed-activation-inc", layer_features)
        state[9] = self._get_sensitivity("q-mixed-activation-dec", layer_features)
        state[10] = self._get_sensitivity("q-mixed-weight-inc", layer_features)
        state[11] = self._get_sensitivity("q-mixed-weight-dec", layer_features)
        state[12] = self._get_sensitivity("q-int8", layer_features)
        state[13] = self._get_sensitivity("q-fp32", layer_features)
        state[14] = last_action.activation
        state[15] = last_action.weight

        state = state.reshape((1, -1))
        self._standard_scaler.partial_fit(state)
        state = self._standard_scaler.transform(state)
        return state.reshape((-1,))

    def _create_policy_with_prediction(self, target_layer, predicted_actions,
                                       reference_policy: CompressionPolicy) -> CompressionPolicy:
        activation_ratio = predicted_actions[0]
        weight_ratio = predicted_actions[1]
        self._last_ratios[target_layer.key] = LastAction(target_layer.key, activation=activation_ratio,
                                                         weight=weight_ratio)
        duplicated_reference = copy.deepcopy(reference_policy)
        selected_method = self._map_to_quantization_action(activation_ratio=activation_ratio,
                                                           weight_ratio=weight_ratio,
                                                           layer_compression_dict=duplicated_reference.layers[
                                                               target_layer.key])
        duplicated_reference.activate_compression(target_layer.key, selected_method)
        return duplicated_reference

    def _map_to_quantization_action(self, activation_ratio, weight_ratio,
                                    layer_compression_dict: dict[str, LayerCompressionSpecification]):
        # biggest compression action wins
        if activation_ratio > self._cfg.mixed_threshold or weight_ratio > self._cfg.mixed_threshold:
            # set and rescale
            if "q-mixed" in layer_compression_dict:
                weight = layer_compression_dict["q-mixed"].parameter_by_key("weight")
                weight.compression_ratio = self._rescale_action(weight_ratio, self._cfg.mixed_threshold)
                activation = layer_compression_dict["q-mixed"].parameter_by_key("activation")
                activation.compression_ratio = self._rescale_action(activation_ratio, self._cfg.mixed_threshold)
                return "q-mixed"

        if activation_ratio > self._cfg.int8_threshold or weight_ratio > self._cfg.int8_threshold:
            if "q-int8" in layer_compression_dict:
                return "q-int8"

        return "q-fp32"

    @staticmethod
    def _rescale_action(ratio, threshold):
        # rescale(e.g.[0.4, 1.0] -> [0.0, 1.0]
        return np.clip((ratio - threshold) / (1.0 - threshold), 0.0, 1.0)

    def get_search_id(self):
        return f"q_it_{self._reward_function.get_search_id()}"


class IndependentSingleLayerQuantizationAgent(IterativeSingleLayerQuantizationAgent):
    STATE_SHAPE = (32,)

    def alg_mode(self) -> AlgorithmicMode:
        return AlgorithmicMode.INDEPENDENT

    def _get_state_shape(self):
        return self.STATE_SHAPE

    def _extract_layer_state(self, target_layer, features):
        state, _ = self._construct_basic_state(features, target_layer)

        state = state.reshape((1, -1))
        self._standard_scaler.partial_fit(state)
        state = self._standard_scaler.transform(state)
        return state.reshape((-1,))

    def _construct_basic_state(self, features, target_layer):
        layer_features = features.features[target_layer.key]
        previous_action = self._get_prev_action(target_layer.key)
        net_bops = self._acc_bops(0, len(features.features), features)
        """
                state features, shape=(11,):
                - [ 0] layer idx                                 (x)
                - [ 1] BOPs layer                                (x)
                - [ 2] n: number of output channels              (x)
                - [ 3] c: number of input channels               (x)
                - [ 4] k: kernel size                            (x)
                - [ 5] s: stride                                 (x)
                - [ 6] n: number of parameters                   (x)
                - [ 6] relative acc. BOPs before layer           (x)
                - [ 7] relative acc. BOPs following the layer    (x)
                - [ 8:18] Sens. mixed activation 0 - 9           (x)
                - [18:28] Sens. mixed weight 0 - 9               (x)
                - [28] Sens. fp32                                (x)
                - [29] Sens. int8                                (x)
                - [30] Current activation ratio                  (x)
                - [31] Current weight ratio                      (x)
    
                (x) = standardized features
            """
        state = np.zeros(self.STATE_SHAPE)
        state[0] = target_layer.step
        state[1] = layer_features.metrics["BOPs"]
        state[2] = layer_features.metrics["out_channels"]
        state[3] = layer_features.metrics["in_channels"]
        state[4] = layer_features.metrics["kernel_size"][0]
        state[5] = layer_features.metrics["stride"][0]
        state[6] = layer_features.metrics["params"]
        state[6] = self._acc_bops(0, target_layer.step, features) / net_bops
        state[7] = self._acc_bops(target_layer.step + 1, len(features.features), features) / net_bops
        state[8:18] = self._get_sensitivity_slice("q-mixed-activation", 10, layer_features)
        state[18:28] = self._get_sensitivity_slice("q-mixed-weight", 10, layer_features)
        state[28] = self._get_sensitivity("q-int8", layer_features)
        state[29] = self._get_sensitivity("q-fp32", layer_features)
        state[30] = previous_action.activation
        state[31] = previous_action.weight
        return state, previous_action

    def _get_prev_action(self, key):
        keys = list(self._last_ratios.keys())
        if key in keys:
            cur_idx = keys.index(key)
            if cur_idx > 0:
                return self._last_ratios[keys[cur_idx - 1]]
        return LastAction(key)

    def get_search_id(self):
        return f"q_ind_{self._reward_function.get_search_id()}"


class Q3AIndependent(IndependentSingleLayerQuantizationAgent):
    Q3A_STATE_SHAPE = (33,)

    def _get_state_shape(self):
        return self.Q3A_STATE_SHAPE

    def _get_num_actions(self):
        return 3

    def _construct_basic_state(self, target_layer, features):
        basic_state, prev_action = super(Q3AIndependent, self)._construct_basic_state(target_layer, features)
        q3a_state = np.zeros(self.Q3A_STATE_SHAPE)
        q3a_state[0:32] = basic_state
        q3a_state[32] = prev_action.threshold_ratio
        return q3a_state, prev_action

    def _create_policy_with_prediction(self, target_layer, predicted_actions,
                                       reference_policy: CompressionPolicy) -> CompressionPolicy:
        activation_ratio = predicted_actions[0]
        weight_ratio = predicted_actions[1]
        threshold_ratio = predicted_actions[2]
        self._last_ratios[target_layer.key] = LastAction(target_layer.key, activation=activation_ratio,
                                                         weight=weight_ratio, threshold_ratio=threshold_ratio)

        duplicated_reference = copy.deepcopy(reference_policy)
        layer_compression_dict = duplicated_reference.layers[target_layer.key]
        selected_method = self._select_quantization_method(threshold_ratio=threshold_ratio,
                                                           layer_compression_dict=layer_compression_dict)
        duplicated_reference.activate_compression(target_layer.key, selected_method)

        if selected_method == "q-mixed":
            self._set_mixed_ratios(activation_ratio, weight_ratio, layer_compression_dict)

        return duplicated_reference

    @staticmethod
    def _set_mixed_ratios(activation_ratio, weight_ratio, layer_compression_dict):
        weight = layer_compression_dict["q-mixed"].parameter_by_key("weight")
        weight.compression_ratio = weight_ratio
        activation = layer_compression_dict["q-mixed"].parameter_by_key("activation")
        activation.compression_ratio = activation_ratio

    def _select_quantization_method(self, threshold_ratio,
                                    layer_compression_dict: dict[str, LayerCompressionSpecification]):
        # biggest compression action wins
        if threshold_ratio > self._cfg.mixed_threshold:
            # set and rescale
            if "q-mixed" in layer_compression_dict:
                return "q-mixed"

        if threshold_ratio > self._cfg.int8_threshold:
            if "q-int8" in layer_compression_dict:
                return "q-int8"

        return "q-fp32"

    def get_search_id(self):
        return f"q3a_ind_{self._reward_function.get_search_id()}"
