import abc
from abc import abstractmethod

import torch

from runtime.compress.compression_policy import CompressionProtocolEntry
from runtime.model.model_handle import AExecutableModel


class AModelEvaluator(metaclass=abc.ABCMeta):

    @abstractmethod
    def retrain(self, executable_model: AExecutableModel):
        pass

    @abstractmethod
    def evaluate(self, executable_model: AExecutableModel, compression_protocol: list[CompressionProtocolEntry]) -> \
            dict[str, float]:
        pass

    @abstractmethod
    def sample_log_probabilities(self, executable_model: AExecutableModel) -> torch.Tensor:
        """
        Predicts probabilities for a preconfigured sample data and returns as numpy array (Result after log_softmax).
        """
        pass

    
class ALatencyEvaluator(metaclass=abc.ABCMeta):

    @abstractmethod
    def measure_latency(self, executable_model: AExecutableModel) -> dict[str, float]:
        pass
