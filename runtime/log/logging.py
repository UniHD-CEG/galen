import pickle
from collections import namedtuple
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import wandb
from matplotlib import pyplot as plt

from runtime.compress.compression_policy import CompressionPolicy, CompressionProtocolEntry
from runtime.log.plot import pq_discrete_plot


class EpisodeData:
    def __init__(self, episode_number: int):
        self._episode_number = episode_number
        self._episode_metrics = dict()
        self._policy = None
        self._rewards = None
        self._compression_protocol = None
        self._step_evaluations = list()
        self._step_actions = list()

        self._start_time = self._timestamp_now()
        self._end_time = None

    @staticmethod
    def _timestamp_now():
        return datetime.now(timezone.utc)

    @property
    def start_time(self):
        return self._start_time

    @property
    def end_time(self):
        return self._end_time

    @property
    def episode_number(self):
        return self._episode_number

    @property
    def predicted_policy(self) -> CompressionPolicy:
        return self._policy

    @property
    def compression_protocol(self) -> list[CompressionProtocolEntry]:
        return self._compression_protocol

    @property
    def metrics(self):
        return self._episode_metrics

    @predicted_policy.setter
    def predicted_policy(self, policy: CompressionPolicy):
        self._policy = policy

    @compression_protocol.setter
    def compression_protocol(self, compression_protocol: list[CompressionProtocolEntry]):
        self._compression_protocol = compression_protocol

    @property
    def rewards(self):
        return self._rewards

    @rewards.setter
    def rewards(self, rewards: np.ndarray):
        self._rewards = rewards

    @property
    def step_evaluations(self):
        return self._step_evaluations

    @property
    def step_actions(self):
        return self._step_actions

    def episode_ended(self):
        self._end_time = self._timestamp_now()

    def __setitem__(self, key, value):
        self._episode_metrics[key] = value

    def __getitem__(self, key):
        return self._episode_metrics[key]

    def __contains__(self, key) -> bool:
        return key in self._episode_metrics

    def append(self, metric_dict):
        self._episode_metrics = self._episode_metrics | metric_dict

    def to_dict(self) -> dict:
        return vars(self)

    def append_step(self, evaluation: dict[str, float], raw_actions: np.ndarray):
        self._step_evaluations.append(evaluation)
        self._step_actions.append(raw_actions)


class LoggingService:

    def __init__(self, search_identifier: str, log_dir: str = "./logs", acceptance_variance=0.05):
        self._current_episode: EpisodeData = None
        self._initial_evaluation = dict()

        self._log_dir = log_dir
        self._acceptance_variance = acceptance_variance
        self._search_identifier = search_identifier

    def search_started(self, num_episodes: int, initial_evaluation: dict[str, float]):
        self._initial_evaluation = initial_evaluation

    def episode_started(self, episode_number: int):
        self._current_episode = EpisodeData(episode_number)

    def episode_completed(self, predicted_policy: CompressionPolicy, episode_evaluation: dict[str, float],
                          optimization_result: tuple[np.ndarray, namedtuple],
                          compression_protocol: list[CompressionProtocolEntry]):
        self._current_episode.episode_ended()
        self._current_episode.predicted_policy = predicted_policy.get_clean_policy()
        self._current_episode.compression_protocol = compression_protocol
        self._current_episode.append(self._include_ratios(episode_evaluation, prefix="episode"))
        rewards, losses = optimization_result
        self._current_episode.rewards = rewards
        self._current_episode["episode-mean-reward"] = np.mean(rewards)
        self._current_episode["episode-number"] = self._current_episode.episode_number
        for name, value in losses._asdict().items():
            self._current_episode[name] = value

        self._wandb_log(self._current_episode)

        episode_data_dict = self._current_episode.to_dict()
        target_file = Path(
            f"{self._log_dir}/{self._search_identifier}-episode-{self._current_episode.episode_number:03d}.pickle")
        target_file.parent.mkdir(exist_ok=True, parents=True)
        with open(target_file, "wb") as file_handle:
            pickle.dump(episode_data_dict, file_handle, protocol=pickle.HIGHEST_PROTOCOL)

    def step_results(self, step_evaluation: dict[str, float], raw_actions: np.ndarray):
        self._current_episode.append_step(self._include_ratios(step_evaluation, prefix="step"), raw_actions)

    def retrain_epoch_completed(self, batch_losses: np.ndarray, batch_acc: np.ndarray):
        self._add_or_concat(batch_losses, "batch_losses")
        self._add_or_concat(batch_acc, "batch_acc")

    def _add_or_concat(self, metric_array, metric_key):
        if metric_key in self._current_episode:
            self._current_episode[metric_key] = np.concatenate((self._current_episode[metric_key], metric_array),
                                                               axis=0)
        else:
            self._current_episode[metric_key] = metric_array

    def retrain_completed(self, epoch_losses: np.ndarray, epoch_acc: np.ndarray):
        self._current_episode["retrain_epoch_loss"] = epoch_losses
        self._current_episode["retrain_epoch_acc"] = epoch_acc

    def _include_ratios(self, compression_evaluation, prefix):
        evaluation = dict()
        for key, value in compression_evaluation.items():
            evaluation[f"{prefix}-{key}"] = value
            if key in self._initial_evaluation:
                evaluation[f"{prefix}-{key}-ratio"] = value / self._initial_evaluation[key]
        return evaluation

    @staticmethod
    def _wandb_log(current_episode: EpisodeData):
        num_steps = len(current_episode.step_evaluations)
        for idx, step_evaluation in enumerate(current_episode.step_evaluations):
            evaluation = step_evaluation
            if idx == (num_steps - 1):
                evaluation = evaluation | current_episode.metrics
                fig = pq_discrete_plot(current_episode.predicted_policy,
                                       current_episode.compression_protocol,
                                       figsize=(20, 10),
                                       title=f"Episode {current_episode.episode_number}",
                                       legend_loc="upper center")
                evaluation["policy"] = wandb.Image(fig)
                plt.close(fig)
            evaluation["step-reward"] = current_episode.rewards[idx]
            evaluation["step-action"] = current_episode.step_actions[idx]
            wandb.log(evaluation)
