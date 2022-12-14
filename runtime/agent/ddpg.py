from collections import namedtuple

import numpy as np
import torch.optim
from scipy import stats
from torch import nn


class DdpgActor(nn.Module):
    def __init__(self,
                 number_input_features=11,
                 number_actions=1,
                 init_bounds=tuple([-3e-3, 3e-3]),
                 **kwargs
                 ):
        super(DdpgActor, self).__init__()
        self.layers = nn.ModuleList()
        number_input = number_input_features
        for i in range(1, 20):
            if (key := f"features_hidden_{i}") in kwargs:
                self.layers.extend([
                    nn.Linear(in_features=number_input, out_features=kwargs[key]),
                    nn.ReLU()
                ])
                number_input = kwargs[key]
        self.out_layer = nn.Linear(in_features=number_input, out_features=number_actions)
        self.out_activation = nn.Sigmoid()
        nn.init.uniform_(self.out_layer.weight, *init_bounds)

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        logits = self.out_layer(out)
        return self.out_activation(logits)


class DdpgCritic(nn.Module):
    def __init__(self,
                 number_input_features=11,
                 number_actions=1,
                 init_bounds=tuple([-3e-4, 3e-4]),
                 features_hidden_1=400,
                 **kwargs
                 ):
        super(DdpgCritic, self).__init__()
        self.linear1_state = nn.Linear(number_input_features, features_hidden_1)
        self.linear1_actions = nn.Linear(number_actions, features_hidden_1)
        self.relu = nn.ReLU()
        self.layers = nn.ModuleList()
        number_input = features_hidden_1
        for i in range(2, 20):
            if (key := f"features_hidden_{i}") in kwargs:
                self.layers.extend([
                    nn.Linear(in_features=number_input, out_features=kwargs[key]),
                    nn.ReLU()
                ])
                number_input = kwargs[key]
        self.out_layer = nn.Linear(number_input, number_actions)
        nn.init.uniform_(self.out_layer.weight, *init_bounds)

    def forward(self, x):
        state, actions = x
        out = self.linear1_state(state) + self.linear1_actions(actions)
        out = self.relu(out)
        for layer in self.layers:
            out = layer(out)
        logits = self.out_layer(out)
        return logits


class DdpgAgentHparams:
    def __init__(self, **kwargs):
        self.tau = float(kwargs.pop("agent_tau", 0.001))
        self.discount = float(kwargs.pop("agent_discount", 0.99))
        self.lr_actor = float(kwargs.pop("lr_actor", 1e-4))
        self.lr_critic = float(kwargs.pop("lr_critic", 1e-3))
        self.critic_decay = float(kwargs.pop("critic_decay", 1e-2))
        self.action_bounds = kwargs.pop("action_bounds", tuple([0.0, 1.0]))
        self.noise_std_decay = float(kwargs.pop("noise_std_decay", 0.95))
        self.noise_initial_std = float(kwargs.pop("noise_initial_std", 0.5))
        self.is_moving_reward_norm = kwargs.pop("enable_moving_reward_normalization", "True") == "True"
        self.normalize_rewards_alpha = float(kwargs.pop("normalize_rewards_alpha", 0.5))


DdpgLosses = namedtuple("DdpgLosses", "critic_loss, actor_loss")


class DdpgAgent:
    """Reusable implementation of a ddpg agent/model"""

    def __init__(self,
                 target_device: torch.device,
                 ddpg_hparams: DdpgAgentHparams = DdpgAgentHparams(),
                 number_input_features: int = 11,
                 number_actions: int = 1,
                 layers: dict[str, int] = None):
        self._hp = ddpg_hparams
        self.target_device = target_device
        self.number_input_features = number_input_features
        self.number_actions = number_actions

        net_cfg = {
            'number_input_features': self.number_input_features,
            'number_actions': self.number_actions
        }
        if layers:
            net_cfg.update(layers)
        else:
            net_cfg.update({
                'features_hidden_1': 400,
                'features_hidden_2': 300
            })

        self.actor = DdpgActor(**net_cfg)
        self.actor_target = DdpgActor(**net_cfg)
        self._hard_update(self.actor, self.actor_target)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self._hp.lr_actor)

        self.critic = DdpgCritic(**net_cfg)
        self.critic_target = DdpgCritic(**net_cfg)
        self._hard_update(self.critic, self.critic_target)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self._hp.lr_critic,
                                                 weight_decay=self._hp.critic_decay)
        self.critic_criterion = nn.MSELoss()

        self.moving_reward_average = None
        self.to(self.target_device)

    def predict(self, episode: int, state: np.ndarray):
        """Expects episodes corrected / relative to episodes predicted by real agens"""
        self.actor.eval()
        with torch.no_grad():
            x = torch.from_numpy(state).view(1, -1).to(self.target_device, dtype=torch.float32)
            raw_actions = self.actor(x).squeeze(0).cpu().detach().numpy()

            decayed_std = self._hp.noise_initial_std * (self._hp.noise_std_decay ** episode)
            # exploration noise: sampling from TruncatedNormal with raw actions as mean and std decayed rel. to episode
            return self._sample_tn(raw_actions, decayed_std)

    def _sample_tn(self, raw_actions_mu, current_std):
        a_array = (self._hp.action_bounds[0] - raw_actions_mu) / current_std
        b_array = (self._hp.action_bounds[1] - raw_actions_mu) / current_std
        std_array = np.full_like(raw_actions_mu, current_std)
        return stats.truncnorm.rvs(a=a_array,
                                   b=b_array,
                                   loc=raw_actions_mu,
                                   scale=std_array,
                                   size=raw_actions_mu.size)

    def optimize(self, experience_batch: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> DdpgLosses:
        state_batch, action_batch, reward_batch, next_state_batch = self._to_tensors(experience_batch)

        y = self._compute_targets(next_state_batch, reward_batch)

        critic_loss = self._update_critic(action_batch, state_batch, y)
        actor_loss = self._update_actor(state_batch)

        self._soft_update(self.critic, self.critic_target)
        self._soft_update(self.actor, self.actor_target)
        return DdpgLosses(critic_loss=critic_loss, actor_loss=actor_loss)

    def _update_actor(self, state_batch):
        self.actor.zero_grad()
        self.actor.train()
        critic_value = self.critic([state_batch, self.actor(state_batch)])
        actor_loss = -torch.mean(critic_value)
        actor_loss.backward()
        self.actor_optimizer.step()
        return actor_loss.cpu().detach().item()

    def _update_critic(self, action_batch, state_batch, y):
        self.critic.zero_grad()
        self.critic.train()
        critic_loss = self.critic_criterion(y, self.critic([state_batch, action_batch]))
        critic_loss.backward()
        self.critic_optimizer.step()
        return critic_loss.cpu().detach().item()

    def _compute_targets(self, next_state_batch, reward_batch):
        with torch.no_grad():
            self.critic_target.eval()
            self.actor_target.eval()
            target_y = reward_batch.unsqueeze(dim=1).expand(
                (-1, self.number_actions)) + self._hp.discount * self.critic_target(
                [next_state_batch, self.actor_target(next_state_batch)])
        return target_y

    def _to_tensors(self, experience_batch: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]):
        state_batch, action_batch, reward_batch, next_state_batch = experience_batch
        reward_batch = self._normalize_reward(reward_batch)

        return torch.from_numpy(state_batch).to(self.target_device, dtype=torch.float32), \
               torch.from_numpy(action_batch).to(self.target_device, dtype=torch.float32), \
               torch.from_numpy(reward_batch).to(self.target_device, dtype=torch.float32), \
               torch.from_numpy(next_state_batch).to(self.target_device, dtype=torch.float32)

    @staticmethod
    def _hard_update(source: nn.Module, target: nn.Module):
        target.load_state_dict(source.state_dict())

    def _soft_update(self, source: nn.Module, target: nn.Module):
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(self._hp.tau * source_param.data + (1 - self._hp.tau) * target_param.data)

    def to(self, target_device: torch.device):
        self.actor.to(target_device)
        self.actor_target.to(target_device)
        self.critic.to(target_device)
        self.critic_target.to(target_device)

    def _normalize_reward(self, reward_batch) -> np.ndarray:
        mean = np.mean(reward_batch)
        if self._hp.is_moving_reward_norm:
            if self.moving_reward_average:
                self.moving_reward_average += self._hp.normalize_rewards_alpha * (
                        mean - self.moving_reward_average)
            else:
                self.moving_reward_average = mean
            mean = self.moving_reward_average
        return reward_batch - mean
