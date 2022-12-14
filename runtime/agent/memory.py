import random

import numpy as np
from numpy_ringbuffer import RingBuffer


class Memory:

    def __init__(self, capacity, action_shape=(1,), obs_shape=(10,)):
        self._obs_buffer = RingBuffer(capacity=capacity, dtype=(float, obs_shape))
        self._action_buffer = RingBuffer(capacity=capacity, dtype=(float, action_shape))
        self._reward_buffer = RingBuffer(capacity=capacity, dtype=float)
        self._next_obs_buffer = RingBuffer(capacity=capacity, dtype=(float, obs_shape))

    def append(self, observation: np.ndarray, action: np.ndarray, reward: np.ndarray, next_observation: np.ndarray):
        self._obs_buffer.append(observation)
        self._action_buffer.append(action)
        self._reward_buffer.append(reward)
        self._next_obs_buffer.append(next_observation)

    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if batch_size > len(self):
            raise Exception("Too less samples in memory to create a batch")
        indices = random.sample(range(len(self)), batch_size)

        return self._obs_buffer[indices], self._action_buffer[indices], self._reward_buffer[indices], \
               self._next_obs_buffer[indices]

    def __len__(self):
        return len(self._obs_buffer)

    def is_empty(self):
        return len(self) == 0
