import sys
from abc import ABCMeta, abstractmethod

import numpy as np

from runtime.controller.algorithmic_mode import AlgorithmicMode


class RewardConfig:
    def __init__(self, **kwargs):
        self.r3_reward_cost_balance_gamma = float(kwargs.pop("r3_reward_cost_balance_gamma", 0.5))
        self.r4_reference_acc = kwargs.pop("r4_reward_ref_acc", None)
        self.r5_beta = float(kwargs.pop("r5_beta", "-0.05"))
        self.r6_beta = float(kwargs.pop("r6_beta", -5.0))
        self.r7_beta = float(kwargs.pop("r7_beta", -5.0))
        self.reward_target_cost_ratio = float(kwargs.pop("reward_target_cost_ratio", 0.5))
        self.episode_cost_key = kwargs.pop("reward_episode_cost_key", "lat") # Has to be overwritten in scripts with "reward_episode_cost_key=BOPs" if latency evaluation is switched off using "enable_latency_eval=False"
        self.step_cost_key = kwargs.pop("reward_step_cost_key", "BOPs")
        self.acc_key = kwargs.pop("reward_acc_key", "acc")
        self.reward = kwargs.pop("reward", "r6")

    def construct_reward(self, alg_mode):
        if self.reward == "r3":
            return ComposedRewardR3(reward_cost_balance_gamma=self.r3_reward_cost_balance_gamma,
                                    mode=alg_mode,
                                    episode_cost_key=self.episode_cost_key,
                                    step_cost_key=self.step_cost_key,
                                    acc_key=self.acc_key)
        if self.reward == "r4":
            return LogCostPositiveRewardR4(mode=alg_mode,
                                           reference_acc=self.r4_reference_acc,
                                           episode_cost_key=self.episode_cost_key,
                                           step_cost_key=self.step_cost_key,
                                           acc_key=self.acc_key)
        if self.reward == "r5":
            return HardExponentialRewardR5(mode=alg_mode,
                                           beta=self.r5_beta,
                                           target_cost_ratio=self.reward_target_cost_ratio,
                                           episode_cost_key=self.episode_cost_key,
                                           step_cost_key=self.step_cost_key,
                                           acc_key=self.acc_key)
        if self.reward == "r6":
            return AbsoluteRewardR6(mode=alg_mode,
                                    beta=self.r6_beta,
                                    target_cost_ratio=self.reward_target_cost_ratio,
                                    episode_cost_key=self.episode_cost_key,
                                    step_cost_key=self.step_cost_key,
                                    acc_key=self.acc_key)

        if self.reward == "r7":
            return ModifiedAbsolutRewardR7(mode=alg_mode,
                                           beta=self.r7_beta,
                                           target_cost_ratio=self.reward_target_cost_ratio,
                                           episode_cost_key=self.episode_cost_key,
                                           step_cost_key=self.step_cost_key,
                                           acc_key=self.acc_key)
        

class AReward(metaclass=ABCMeta):

    def __init__(self, mode: AlgorithmicMode, episode_cost_key="BOPs", step_cost_key="BOPs",
                 acc_key="acc"):
        self._ep_cost_key = episode_cost_key
        self._step_cost_key = step_cost_key
        self._acc_key = acc_key
        self._alg_mode = mode
        self._reference_evaluation = None
        self._last_episode_evaluation = None

    def initialize(self, reference_evaluation: dict[str, float]):
        self._reference_evaluation = reference_evaluation

    def _initial_step_term_last_episode_cost(self):
        if self._alg_mode == AlgorithmicMode.ITERATIVE and self._last_episode_evaluation:
            return self._last_episode_evaluation[self._step_cost_key]
        return self._reference_evaluation[self._step_cost_key]

    @abstractmethod
    def requires_step_eval(self) -> bool:
        pass

    @abstractmethod
    def compute_rewards(self, episode_evaluation, step_evaluation) -> np.ndarray:
        pass

    @abstractmethod
    def get_search_id(self):
        pass


class ComposedRewardR3(AReward):

    def __init__(self, reward_cost_balance_gamma: float, mode: AlgorithmicMode, **kwargs):
        super(ComposedRewardR3, self).__init__(mode, **kwargs)
        self._gamma = reward_cost_balance_gamma

    def requires_step_eval(self) -> bool:
        return True

    def compute_rewards(self, episode_evaluation, step_evaluations) -> np.ndarray:
        num_steps = len(step_evaluations)
        r_cost = np.zeros((num_steps,))
        r_acc = np.zeros((num_steps,))

        step_term_last_episode_cost = self._initial_step_term_last_episode_cost()
        last_step_cost = step_term_last_episode_cost

        for i in range(num_steps):
            cost_ep_term = episode_evaluation[self._ep_cost_key] / self._reference_evaluation[self._ep_cost_key]
            cost_step_reference = (
                    step_term_last_episode_cost - episode_evaluation[self._step_cost_key] + sys.float_info.epsilon)
            cost_step_term = (last_step_cost - step_evaluations[i][self._step_cost_key]) / cost_step_reference
            r_cost[i] = - (cost_ep_term - cost_step_term)
            last_step_cost = step_evaluations[i][self._step_cost_key]

            acc_ep_term = 1 - episode_evaluation[self._acc_key] / self._reference_evaluation[self._acc_key]
            acc_step_term = (1 - step_evaluations[i][self._acc_key] / self._reference_evaluation[
                self._acc_key]) / num_steps
            r_acc[i] = - (acc_ep_term + acc_step_term)

        self._last_episode_evaluation = episode_evaluation

        return self._gamma * r_cost + (1 - self._gamma) * r_acc

    def get_search_id(self):
        return f"r3_g{self._gamma}_c{self._ep_cost_key}"


class LogCostPositiveRewardR4(AReward):

    def __init__(self, mode: AlgorithmicMode, reference_acc=None, **kwargs):
        super(LogCostPositiveRewardR4, self).__init__(mode, **kwargs)
        self._reference_acc = reference_acc

    def requires_step_eval(self) -> bool:
        return False

    def compute_rewards(self, episode_evaluation, step_evaluation) -> np.ndarray:
        cost_ratio = np.full((len(step_evaluation),),
                             episode_evaluation[self._ep_cost_key] / self._reference_evaluation[self._ep_cost_key])

        return -np.log(cost_ratio) * episode_evaluation[self._acc_key] / self._get_reference_acc()

    def _get_reference_acc(self) -> float:
        if self._reference_acc:
            return float(self._reference_acc)
        return self._reference_evaluation[self._acc_key]

    def get_search_id(self):
        return f"r4_reg{self._reference_acc}_c{self._ep_cost_key}"


class HardExponentialRewardR5(AReward):

    def __init__(self, mode: AlgorithmicMode, beta, target_cost_ratio, **kwargs):
        super(HardExponentialRewardR5, self).__init__(mode, **kwargs)
        self._beta = beta
        self._cost_ratio = target_cost_ratio

    def requires_step_eval(self) -> bool:
        return False

    def compute_rewards(self, episode_evaluation, step_evaluation) -> np.ndarray:
        target_cost = self._reference_evaluation[self._ep_cost_key] * self._cost_ratio
        achieved_cost = episode_evaluation[self._ep_cost_key]
        quality = episode_evaluation[self._acc_key]

        if achieved_cost <= target_cost:
            reward = quality
        else:
            reward = quality * ((achieved_cost / target_cost) ** self._beta)
        return np.full((len(step_evaluation),), reward)

    def get_search_id(self):
        return f"r5_b{self._beta}_r{self._cost_ratio}_c{self._ep_cost_key}"


class AbsoluteRewardR6(AReward):

    def __init__(self, mode: AlgorithmicMode, beta, target_cost_ratio, **kwargs):
        super(AbsoluteRewardR6, self).__init__(mode, **kwargs)
        self._beta = beta
        self._cost_ratio = target_cost_ratio

    def requires_step_eval(self) -> bool:
        return False

    def compute_rewards(self, episode_evaluation, step_evaluation) -> np.ndarray:
        target_cost = self._reference_evaluation[self._ep_cost_key] * self._cost_ratio
        achieved_cost = episode_evaluation[self._ep_cost_key]
        quality = episode_evaluation[self._acc_key]

        reward = quality + self._beta * abs((achieved_cost / target_cost) - 1)
        return np.full((len(step_evaluation),), reward)

    def get_search_id(self):
        return f"r6_b{self._beta}_r{self._cost_ratio}_c{self._ep_cost_key}"


class ModifiedAbsolutRewardR7(AReward):

    def __init__(self, mode: AlgorithmicMode, beta, target_cost_ratio, **kwargs):
        super(ModifiedAbsolutRewardR7, self).__init__(mode, **kwargs)
        self._beta = beta
        self._cost_ratio = target_cost_ratio

    def requires_step_eval(self) -> bool:
        return False

    def compute_rewards(self, episode_evaluation, step_evaluation) -> np.ndarray:
        target_cost = self._reference_evaluation[self._ep_cost_key] * self._cost_ratio
        achieved_cost = episode_evaluation[self._ep_cost_key]
        quality = episode_evaluation[self._acc_key]

        if achieved_cost <= target_cost:
            reward = quality
        else:
            reward = quality + self._beta * abs((achieved_cost / target_cost) - 1)
        return np.full((len(step_evaluation),), reward)

    def get_search_id(self):
        return f"r7_b{self._beta}_r{self._cost_ratio}_c{self._ep_cost_key}"
