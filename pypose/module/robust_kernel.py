import torch
from torch import nn, Tensor


class Huber(nn.Module):
    r"""The robust Huber kernel function that is less sensitive to outliers than squared error
    in a non-linear optimization problem.
    See `Huber loss <https://en.wikipedia.org/wiki/Huber_loss>`_ for more information.

    .. math::
        \bm{y}_i = \begin{cases}
            0.5 \bm{x}_i^2                       & \text{if } |\bm{x}_i| < \delta \\
            \delta * (|\bm{x}_i| - 0.5 * \delta) & \text{otherwise }
        \end{cases},

    where :math:`\delta` (delta) is a threshold, :math:`\bm{x}` and :math:`\bm{y}` are the input
    and output tensors, respectively.

    Args:
        delta (float, optional): Specify the threshold at which to scale the input. The value must
            be positive.  Default: 1.0

    Note:
        The output tensor has the same shape with the input tensor. Use `torch.nn.HuberLoss
        <https://pytorch.org/docs/stable/generated/torch.nn.HuberLoss.html>`_ instead, if a scalar
        Huber loss function is needed.

    Example:
        >>> kernel = pp.module.Huber()
        >>> input = torch.randn(3).abs()
        tensor([1.9087, 0.2256, 0.6565])
        >>> output = kernel(input)
        tensor([1.4087, 0.0254, 0.2155])
    """
    def __init__(self, delta: float = 1.0) -> None:
        super().__init__()
        assert delta > 0, ValueError("Invalid delta value: {}".format(delta))
        self.delta = delta

    def forward(self, input: Tensor) -> Tensor:
        ''''''
        mask = input.abs() < self.delta
        output = torch.zeros_like(input)
        output[mask] = (input[mask]**2)/2
        output[~mask] = self.delta * (input[~mask].abs() - 0.5 * self.delta)
        return output


class PseudoHuber(nn.Module):
    r"""The robust Pseudo Huber kernel function that is less sensitive to outliers than squared
    error in a non-linear optimization problem. It approximates :math:`\bm{x}^2/2` for small
    values of :math:`\bm{x}`, and approximates a straight line with slope :math:`\delta` for large
    values of :math:`\bm{x}`.

    .. math::
        \bm{y}_i = \delta^2 (\sqrt{1 + (\bm{x}/\delta)^2} - 1)

    where :math:`\delta` (delta) defines the steepness of the slope, :math:`\bm{x}` and
    :math:`\bm{y}` are the input and output tensors, respectively.  It can be used as a smooth
    version of :obj:`Huber`.

    Args:
        delta (float, optional): Specify the slope. The value must be positive.  Default: 1.0

    Note:
        The output tensor has the same shape with the input tensor.

    Example:
        >>> kernel = pp.module.PseudoHuber()
        >>> input = torch.randn(3).abs()
        tensor([0.3256, 0.9250, 0.2337])
        >>> output = kernel(input)
        tensor([0.0517, 0.3622, 0.0269])
    """
    def __init__(self, delta: float = 1.0) -> None:
        super().__init__()
        assert delta > 0, ValueError("Invalid delta value: {}".format(delta))
        self.delta = delta

    def forward(self, input: Tensor) -> Tensor:
        ''''''
        return self.delta**2 * ((1 + (input/self.delta)**2).sqrt() - 1)
