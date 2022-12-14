import abc
import math
from abc import abstractmethod, ABCMeta
from collections import namedtuple
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from runtime.agent.ddpg import DdpgAgentHparams, DdpgAgent, DdpgLosses
from runtime.agent.memory import Memory
from runtime.agent.reward import RewardConfig
from runtime.compress.compression_policy import CompressionPolicy
from runtime.controller.algorithmic_mode import AlgorithmicMode
from runtime.feature_extraction.feature_extractor import ModelFeatures, LayerFeatures

GlobalStep = namedtuple("GlobalStep", "step, episode")


class AAgent(metaclass=abc.ABCMeta):

    @abstractmethod
    def initialize(self, reference_evaluation: dict[str, float]):
        pass

    @abstractmethod
    def do_prediction(self, global_step: GlobalStep, features: ModelFeatures,
                      last_policy: CompressionPolicy) -> tuple[CompressionPolicy | None, np.ndarray | None]:
        pass

    @abstractmethod
    def pass_evaluation_results(self, global_step: GlobalStep, compression_evaluation: dict[str, float],
                                next_observation: ModelFeatures):
        pass

    @abstractmethod
    def pass_episode_results(self, episode: int, compression_evaluation: dict[str, float],
                             final_observation: ModelFeatures) -> tuple[np.ndarray, namedtuple]:
        pass

    @abstractmethod
    def config(self) -> dict:
        pass

    @abstractmethod
    def supported_method(self) -> list[str]:
        pass

    @abstractmethod
    def alg_mode(self) -> AlgorithmicMode:
        pass

    @abstractmethod
    def requires_step_eval(self) -> bool:
        pass

    def config_overrides(self) -> dict[str, str]:
        return dict()

    @abstractmethod
    def get_search_id(self):
        pass


class AgentWrapper:

    def __init__(self, agent: AAgent):
        self._agent = agent

    def predict_policy(self, global_step: GlobalStep, environment_observation: ModelFeatures,
                       last_policy: CompressionPolicy) -> tuple[CompressionPolicy | None, np.ndarray | None]:
        predicted_policy = self._agent.do_prediction(global_step, environment_observation, last_policy)
        return predicted_policy

    def pass_step_results(self, global_step: GlobalStep, compression_evaluation: dict[str, float],
                          next_observation: ModelFeatures):
        self._agent.pass_evaluation_results(global_step, compression_evaluation, next_observation)

    def pass_episode_results(self, episode: int, compression_evaluation: dict[str, float],
                             final_observation: ModelFeatures) -> tuple[np.ndarray, namedtuple]:
        optimization_result = self._agent.pass_episode_results(episode, compression_evaluation, final_observation)
        return optimization_result

    def initialize(self, reference_evaluation: dict[str, float]):
        self._agent.initialize(reference_evaluation)

    def requires_step_eval(self):
        return self._agent.requires_step_eval()


@dataclass
class StepProtocol:
    step: int
    observation: np.ndarray
    action: np.ndarray
    next_observation: np.ndarray = None
    evaluation: dict[str, float] = None
    delta_macs: int = 0


TargetLayer = namedtuple("TargetLayer", "step, key, method_key")


class GenericAgentConfig(DdpgAgentHparams):
    def __init__(self, **kwargs):
        super(GenericAgentConfig, self).__init__(**kwargs)
        self.experience_buffer_size = int(kwargs.pop("experience_buffer_size", 2000))
        self.agent_batch_size = int(kwargs.pop("agent_batch_size", 128))
        self.warmup_episodes = int(kwargs.pop("warmup_episodes", 10))
        self.agent_action_seed = kwargs.pop("agent_seed", None)
        self.reward_config = RewardConfig(**kwargs)
        self.reward_config_dict = vars(self.reward_config)


class ADdpgAgent(AAgent, metaclass=ABCMeta):
    NOT_ALLOWED_SENS = 1000.0

    def __init__(self, target_device: torch.device, agent_config=GenericAgentConfig()):
        self._cfg = agent_config
        self._num_actions = self._get_num_actions()
        self._state_shape = self._get_state_shape()
        self._ddpg_agent = DdpgAgent(target_device, self._cfg, number_input_features=self._state_shape[0],
                                     number_actions=self._num_actions, layers=self._get_custom_hidden_layers())
        self._episode_protocol = list()
        self._current_episode_step: StepProtocol = None
        self._experience_buffer = Memory(capacity=self._cfg.experience_buffer_size, obs_shape=self._state_shape,
                                         action_shape=(self._num_actions,))
        self._min_max_scaler = MinMaxScaler()
        self._standard_scaler = StandardScaler()
        self._reward_function = self._cfg.reward_config.construct_reward(self.alg_mode())
        if self._cfg.agent_action_seed:
            self._random_gen = np.random.Generator(np.random.PCG64(int(self._cfg.agent_action_seed)))
        else:
            self._random_gen = np.random.Generator(np.random.PCG64())

    def _get_custom_hidden_layers(self):
        return None

    @abstractmethod
    def _get_state_shape(self):
        pass

    @abstractmethod
    def _get_num_actions(self):
        pass

    def requires_step_eval(self) -> bool:
        return self._reward_function.requires_step_eval()

    def alg_mode(self) -> AlgorithmicMode:
        return AlgorithmicMode.ITERATIVE

    def initialize(self, reference_evaluation: dict[str, float]):
        self._reward_function.initialize(reference_evaluation)

    def do_prediction(self, global_step: GlobalStep, features: ModelFeatures,
                      last_policy: CompressionPolicy) -> tuple[CompressionPolicy | None, np.ndarray | None]:
        target_layer = self._resolve_target_layer(global_step.step, last_policy)
        if target_layer is None:
            # pass layer without predicting an action - pruning is not supported for this layer
            return None, None

        layer_state = self._extract_layer_state(target_layer, features)
        predicted_actions = self._predict_actions(global_step, layer_state)
        predicted_policy = self._create_policy_with_prediction(target_layer, predicted_actions, last_policy)

        if self._current_episode_step:
            # next_observation for last executed step
            self._current_episode_step.next_observation = layer_state

        self._current_episode_step = StepProtocol(step=global_step.step, observation=layer_state,
                                                  action=predicted_actions)
        self._episode_protocol.append(self._current_episode_step)
        return predicted_policy, predicted_actions

    def pass_evaluation_results(self, global_step: GlobalStep, step_evaluation: dict[str, float],
                                next_observation: ModelFeatures):
        if self._current_episode_step:
            self._current_episode_step.evaluation = step_evaluation

    def pass_episode_results(self, episode: int, episode_evaluation: dict[str, float],
                             final_observation: ModelFeatures) -> tuple[np.ndarray, namedtuple]:

        rewards = self._reward_function.compute_rewards(episode_evaluation,
                                                        [step_eval.evaluation for step_eval in self._episode_protocol])
        for step_idx, protocol in enumerate(self._episode_protocol):
            next_observation = protocol.next_observation
            if next_observation is None:
                # for last layer use original observation as last observation, same for: [AMC, HAQ]
                next_observation = protocol.observation
            self._append_to_experience_buffer(next_observation, protocol, rewards, step_idx)

        # reset protocol
        self._episode_protocol = list()
        self._pre_optimize()
        losses = self._optimize(episode)
        return rewards, losses

    def _append_to_experience_buffer(self, next_observation, protocol, rewards, step_idx):
        if not (self._experience_buffer.is_empty() and step_idx == 0):
            # skip first transition due to invalid state
            self._experience_buffer.append(protocol.observation, protocol.action, rewards[step_idx], next_observation)

    def _resolve_target_layer(self, step: int, reference_policy: CompressionPolicy) -> TargetLayer | None:
        target_layer_key = list(reference_policy.layers.keys())[step]
        for method_key, method_spec in reference_policy.layers[target_layer_key].items():
            if method_key in self.supported_method():
                # return first one found
                return TargetLayer(step=step, key=target_layer_key, method_key=method_key)
        return None

    @abstractmethod
    def _extract_layer_state(self, target_layer, features):
        pass

    def _predict_actions(self, global_step: GlobalStep, layer_state: np.ndarray) -> np.ndarray:
        if global_step.episode >= self._cfg.warmup_episodes:
            # correct episode number to not include warm up into noise variance decay
            prediction_episode = global_step.episode - self._cfg.warmup_episodes
            return self._ddpg_agent.predict(prediction_episode, layer_state)
        else:
            return self._random_gen.uniform(self._cfg.action_bounds[0], self._cfg.action_bounds[1],
                                            (self._num_actions,))

    def _optimize(self, episode) -> DdpgLosses:
        if episode >= self._cfg.warmup_episodes:
            if len(self._experience_buffer) > self._cfg.agent_batch_size:
                # skip optimization if not enough transitions were collected
                experience_batch = self._experience_buffer.sample(self._cfg.agent_batch_size)
                return self._ddpg_agent.optimize(experience_batch)
        return DdpgLosses(math.nan, math.nan)

    @abstractmethod
    def _create_policy_with_prediction(self, target_layer, predicted_actions, reference_policy) -> CompressionPolicy:
        pass

    def config(self) -> dict:
        return vars(self._cfg)

    def _pre_optimize(self):
        pass

    @staticmethod
    def _acc_single_metric(start: int, stop: int, metric_name, features: ModelFeatures):
        acc = 0
        for layer_features in list(features.features.values())[start:stop]:
            acc += layer_features.metrics[metric_name]
        return acc

    def _acc_macs(self, start: int, stop: int, features: ModelFeatures):
        return self._acc_single_metric(start, stop, "MACs", features)

    def _acc_bops(self, start: int, stop: int, features: ModelFeatures):
        return self._acc_single_metric(start, stop, "BOPs", features)

    def _get_sensitivity_slice(self, key: str, count: int, layer_features: LayerFeatures):
        sens_state = np.zeros((count,))
        for idx in range(count):
            sens_state[idx] = self._get_sensitivity(f"{key}-{idx}", layer_features)

        return sens_state

    def _get_sensitivity(self, sens_key, layer_features: LayerFeatures):
        if sens_key in layer_features.sensitivity.policy_results:
            return layer_features.sensitivity.policy_results[sens_key]
        return self.NOT_ALLOWED_SENS
