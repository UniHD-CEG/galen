import numpy as np

from runtime.compress.compress_adapters import ADiscretizer


class InverseRatioDiscretizer(ADiscretizer):
    def discretize_parameter(self, compression_ratio, reference) -> int:
        # c_param is sparsity
        return int(self._discretize(compression_ratio, reference).item())

    @staticmethod
    def _discretize(compression_ratio, reference) -> np.ndarray:
        return np.clip(np.floor((1.0 - compression_ratio) * reference) + 1, 1, reference)


class RoundToDiscretizer(InverseRatioDiscretizer):
    def __init__(self, round_to=16):
        super(RoundToDiscretizer, self).__init__()
        self._round_to = round_to

    def discretize_parameter(self, compression_ratio, reference) -> int:
        # real division is used by purpose -> allow to predict original channel count also if not a multiple of round_to
        reference = reference / self._round_to
        q = self._discretize(compression_ratio, reference)

        return int(q * self._round_to)
