import copy
from dataclasses import dataclass

import numpy as np
import torch

from runtime.agent.quantization_agent import QuantizationAgentConfig, IterativeSingleLayerQuantizationAgent
from runtime.compress.compression_policy import CompressionPolicy
from runtime.controller.algorithmic_mode import AlgorithmicMode
from runtime.feature_extraction.feature_extractor import LayerFeatures


@dataclass
class LastAction:
    sparsity: float = 0.0
    activation: float = 0.0
    weight: float = 0.0


class PruningQuantizationAgentConfig(QuantizationAgentConfig):

    def __init__(self, **kwargs):
        super(PruningQuantizationAgentConfig, self).__init__(**kwargs)
        self.pq_agent_channel_round_to = float(kwargs.pop("pq_agent_channel_round_to", 32))


class PruningQuantizationAgent(IterativeSingleLayerQuantizationAgent):
    # inheritance for better reuse - no logical dep. to quantization agent
    STATE_SHAPE = (47,)

    def __init__(self, target_device: torch.device, agent_config=PruningQuantizationAgentConfig()):
        super(PruningQuantizationAgent, self).__init__(target_device, agent_config)
        self._last_action = LastAction()

    def alg_mode(self) -> AlgorithmicMode:
        return AlgorithmicMode.INDEPENDENT

    def _get_state_shape(self):
        return self.STATE_SHAPE

    def _get_num_actions(self):
        return 3

    def _get_custom_hidden_layers(self):
        # return {
        #     'features_hidden_1': 600,
        #     'features_hidden_2': 400,
        #     'features_hidden_3': 300,
        # }
        return {
            'features_hidden_1': 400,
            'features_hidden_2': 300
        }

    def config_overrides(self) -> dict[str, str]:
        return {"p_channel_round_to": self._cfg.pq_agent_channel_round_to}

    def _extract_layer_state(self, target_layer, features):
        layer_features = features.features[target_layer.key]
        net_bops = self._acc_bops(0, len(features.features), features)
        net_macs = self._acc_macs(0, len(features.features), features)
        pruning_method_key = self._get_pruning_method(layer_features)
        sample_count = 10

        state = np.zeros(self.STATE_SHAPE)
        state[0] = target_layer.step
        state[1] = layer_features.metrics["MACs"]
        state[2] = layer_features.metrics["BOPs"]
        state[3] = layer_features.metrics["out_channels"]
        state[4] = layer_features.metrics["in_channels"]
        state[5] = layer_features.metrics["kernel_size"][0]
        state[6] = layer_features.metrics["stride"][0]
        state[7] = layer_features.metrics["params"]
        state[8] = self._acc_macs(0, target_layer.step, features) / net_macs
        state[9] = self._acc_macs(target_layer.step + 1, len(features.features), features) / net_macs
        state[10] = self._acc_bops(0, target_layer.step, features) / net_bops
        state[11] = self._acc_bops(target_layer.step + 1, len(features.features), features) / net_bops
        state[12:22] = self._get_sensitivity_slice(pruning_method_key, sample_count, layer_features)
        state[22:32] = self._get_sensitivity_slice("q-mixed-activation", sample_count, layer_features)
        state[32:42] = self._get_sensitivity_slice("q-mixed-weight", sample_count, layer_features)
        state[42] = self._get_sensitivity("q-int8", layer_features)
        state[43] = self._get_sensitivity("q-fp32", layer_features)
        state[44] = self._last_action.sparsity
        state[45] = self._last_action.activation
        state[46] = self._last_action.weight

        state = state.reshape((1, -1))
        self._standard_scaler.partial_fit(state)
        state = self._standard_scaler.transform(state)
        return state.reshape((-1,))

    def _create_policy_with_prediction(self, target_layer, predicted_actions, reference_policy) -> CompressionPolicy:
        action = LastAction(predicted_actions[0], predicted_actions[1], predicted_actions[2])
        self._last_action = action
        duplicated_reference = copy.deepcopy(reference_policy)
        layer_compression_dict = duplicated_reference.layers[target_layer.key]

        # pruning
        if any((method_key := method).startswith("p-") for method in layer_compression_dict.keys()):
            # pruning is supported for layer - else ignore pruning action
            compression_parameter = duplicated_reference.layers[target_layer.key][method_key].parameter_by_key(
                "sparsity")
            compression_parameter.compression_ratio = action.sparsity
            duplicated_reference.activate_compression(target_layer.key, method_key)

        # quantization
        selected_method = self._map_to_quantization_action(activation_ratio=action.activation,
                                                           weight_ratio=action.weight,
                                                           layer_compression_dict=layer_compression_dict)
        duplicated_reference.activate_compression(target_layer.key, selected_method)
        return duplicated_reference

    def supported_method(self) -> list[str]:
        return ["q-mixed", "q-fp32", "q-int8", "p-lin", "p-conv"]

    def _pre_optimize(self):
        self._last_action = LastAction()

    def _get_pruning_method(self, layer_features: LayerFeatures):
        if any(method.startswith("p-lin") for method in layer_features.sensitivity.policy_results.keys()):
            return "p-lin-sparsity"
        return "p-conv-sparsity"

    def get_search_id(self):
        return f"pq_{self._reward_function.get_search_id()}"
