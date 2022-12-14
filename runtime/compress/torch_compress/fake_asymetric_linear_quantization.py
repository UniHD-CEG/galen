import torch
import torch.autograd


def determine_quantization_params(n_bits, x_min, x_max):
    # asymmetric linear quantization
    n = 2 ** n_bits - 1
    scale = n / torch.clamp((x_max - x_min), min=1e-8)
    # round zero point to represent x_min as real 0 + we use signed integers
    zero_point = (scale * x_min).round() + 2 ** (n_bits - 1)

    return scale, zero_point


def linear_quantize(x, scale, zero_point, n_bits):
    scale, zero_point = reshape_params(scale, x, zero_point)
    n = 2 ** (n_bits - 1)
    # clamping is required to cut values not representable by integers (but by floats)
    return torch.clamp(torch.round(scale * x - zero_point), -n, n - 1)


def linear_dequantize(x, scale, zero_point):
    scale, zero_point = reshape_params(scale, x, zero_point)
    return (x + zero_point) / scale


def reshape_params(scale, x, zero_point):
    required_shape = [1] * len(x.shape)
    required_shape[0] = -1
    return scale.view(required_shape), zero_point.view(required_shape)


def fake_quantize(x, n_bits, x_min, x_max):
    # borrowed from ZeroQ
    if x_min is None or x_max is None:
        x_min, x_max = x.min(), x.max()

    scale, zero_point = determine_quantization_params(n_bits, x_min, x_max)
    x_quant = linear_quantize(x, scale, zero_point, n_bits)

    # this is fake quantization, we map directly back to original float values for computation
    # however includes losses which would be introduced by a real quantization and integer computations
    return linear_dequantize(x_quant, scale, zero_point)


class FakeQuantizeOp(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, n_bits):
        first_dim = x.shape[0]
        by_output = x.detach().view(first_dim, -1)
        x_min = by_output.min(dim=1)[0]
        x_max = by_output.max(dim=1)[0]
        quantized_weight = fake_quantize(x, n_bits, x_min, x_max)
        return quantized_weight

    @staticmethod
    def backward(ctx, grad_output):
        # straight through estimator
        return grad_output, None, None, None, None
