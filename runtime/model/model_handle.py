from abc import ABCMeta, abstractmethod
from enum import Enum
from typing import Tuple

from runtime.compress.compression_policy import CompressionPolicy


class ModelPhase(Enum):
    VALIDATION = 1
    CALIBRATION = 2


class AModelReference(metaclass=ABCMeta):

    @abstractmethod
    def all_layer_keys(self) -> set[str]:
        pass

    @abstractmethod
    def batch_input_shape(self) -> Tuple[int, ...]:
        pass

    @abstractmethod
    def frozen_layers(self, method_key: str) -> list[str]:
        pass


class AExecutableModel(AModelReference, metaclass=ABCMeta):
    @property
    @abstractmethod
    def applied_policy(self) -> CompressionPolicy:
        pass

    @property
    @abstractmethod
    def compression_protocol(self) -> list[CompressionPolicy]:
        pass

    @compression_protocol.setter
    def compression_protocol(self, protocol):
        pass

    @applied_policy.setter
    @abstractmethod
    def applied_policy(self, policy: CompressionPolicy):
        pass


class AModelFactory(metaclass=ABCMeta):

    @abstractmethod
    def get_reference_model(self):
        pass

    @abstractmethod
    def to_executable_model(self, model_reference: AModelReference) -> AExecutableModel:
        pass
