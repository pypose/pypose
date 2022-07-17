import torch
from torch import nn, Tensor


class Huber(nn.Module):
    r"""The robust Huber kernel cost function.

    .. math::
        \bm{y}_i = \begin{cases}
            \bm{x}_i                            & \text{if } \sqrt{\bm{x}_i} < \delta \\
            2 \delta \sqrt{\bm{x}_i} - \delta^2 & \text{otherwise }
        \end{cases},

    where :math:`\delta` (delta) is a threshold, :math:`\bm{x}` and :math:`\bm{y}` are the input
    and output tensors, respectively.

    Args:
        delta (float, optional): Specify the threshold at which to scale the input. The value must
            be positive. Default: 1.0

    Note:
        The input has to be a non-negative tensor and the output tensor has the same shape with the
        input. Use `torch.nn.HuberLoss
        <https://pytorch.org/docs/stable/generated/torch.nn.HuberLoss.html>`_ instead, if a scalar
        Huber loss function is needed.

    Example:
        >>> import pypose.optim.kernel as ppok
        >>> kernel = ppok.Huber()
        >>> input = torch.tensor([0, 0.5, 1, 2, 3])
        >>> kernel(input)
        tensor([0.0000, 0.5000, 1.0000, 1.8284, 2.4641])
    """
    def __init__(self, delta: float = 1.0) -> None:
        super().__init__()
        assert delta > 0, ValueError("delta has to be positive: {}".format(delta))
        self.delta = delta
        self.delta2 = delta**2

    def forward(self, input: Tensor) -> Tensor:
        '''
        Args:
            input (torch.Tensor): the input tensor (non-negative).
        '''
        assert torch.all(input >= 0), 'input has to be non-negative.'
        mask = input.sqrt() < self.delta
        output = torch.zeros_like(input)
        output[mask] = input[mask]
        output[~mask] = 2 * self.delta * input[~mask].sqrt() - self.delta2
        return output


class PseudoHuber(nn.Module):
    r"""The robust pseudo Huber kernel cost function.

    .. math::
        \bm{y}_i = 2\delta^2 \left(\sqrt{1 + \bm{x}_i/\delta^2} - 1\right),

    where :math:`\delta` (delta) defines the steepness of the slope, :math:`\bm{x}` and
    :math:`\bm{y}` are the input and output tensors, respectively.  It can be used as a smooth
    version of :obj:`Huber`.

    Args:
        delta (float, optional): Specify the slope. The value must be positive. Default: 1.0

    Note:
        The input has to be a non-negative tensor and the output tensor has the same shape with the
        input.

    Example:
        >>> import pypose.optim.kernel as ppok
        >>> kernel = ppok.PseudoHuber()
        >>> input = torch.tensor([0, 0.5, 1, 2, 3])
        >>> kernel(input)
        tensor([0.0000, 0.4495, 0.8284, 1.4641, 2.0000])
    """
    def __init__(self, delta: float = 1.0) -> None:
        super().__init__()
        assert delta > 0, ValueError("delta has to be positive: {}".format(delta))
        self.delta2 = delta**2

    def forward(self, input: Tensor) -> Tensor:
        '''
        Args:
            input (torch.Tensor): the input tensor (non-negative).
        '''
        assert torch.all(input >= 0), 'input has to be non-negative'
        return 2 * self.delta2 * ((input/self.delta2 + 1).sqrt() - 1)


class Cauchy(nn.Module):
    r"""The robust Cauchy kernel cost function.

    .. math::
        \bm{y}_i = \delta^2 \log\left(1 + \frac{\bm{x}_i}{\delta^2}\right),

    where :math:`\delta` (delta) is a hyperparameter, :math:`\bm{x}` and :math:`\bm{y}` are the
    input and output tensors, respectively.

    Args:
        delta (float, optional): Specify the Cauchy cost. The value must be positive. Default: 1.0

    Note:
        The input has to be a non-negative tensor and the output tensor has the same shape with the
        input.

    Example:
        >>> import pypose.optim.kernel as ppok
        >>> kernel = ppok.Cauchy()
        >>> input = torch.tensor([0, 0.5, 1, 2, 3])
        >>> kernel(input)
        tensor([0.0000, 0.4055, 0.6931, 1.0986, 1.3863])
    """
    def __init__(self, delta: float = 1.0) -> None:
        super().__init__()
        assert delta > 0, ValueError("delta has to be positive: {}".format(delta))
        self.delta2 = delta**2

    def forward(self, input: Tensor) -> Tensor:
        '''
        Args:
            input (torch.Tensor): the input tensor (non-negative).
        '''
        assert torch.all(input >= 0), 'input has to be non-negative'
        return self.delta2 * (input/self.delta2 + 1).log()
