import torch
from torch import nn, Tensor
import math
import pypose.optim.kernel as kernal


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



class SoftLOne(nn.Module):
    r"""The robust SoftLOne kernel cost function.

    .. math::
        \bm{y}_i=2\left ( \delta \sqrt{\frac{1}{{\delta{}}^{2}}+\bm{x}_i}- {{\delta{}}^{2}}\right )

    where :math:`\delta` (delta) is a hyperparameter, :math:`\bm{x}` and :math:`\bm{y}` are the
    input and output tensors, respectively.

    Args:
        delta (float, optional): Specify the SoftLOne cost. The value must be positive. Default: 1.0


    Note:
        The input has to be a non-negative tensor and the output tensor has the same shape with the
        input.

    Example:
        >>> import pypose.optim.kernel as ppok
        >>> kernel = ppok.SoftLOne()
        >>> input = torch.tensor([0, 0.5, 1, 2, 3])
        >>> kernel(input)
        tensor([0.0000, 0.4495, 0.8284, 1.4641, 2.0000])
    """
    def __init__(self, delta: float = 1.0) -> None:
        super().__init__()
        assert delta > 0, ValueError("delta has to be positive: {}".format(delta))
        self.delta1 = delta
        self.delta2 = delta**2

    def forward(self, input: Tensor) -> Tensor:
        '''
        Args:
            input (torch.Tensor): the input tensor (non-negative).
        '''
        assert torch.all(input >= 0), 'input has to be non-negative'
        return 2*(self.delta1*(1/self.delta2+input).sqrt()-self.delta2)

class Arctan(nn.Module):
    r"""The robust Arctan kernel cost function.

    .. math::
        \bm{y}_i=\delta ^{2}\arctan \left ( \frac{\bm{x}_i}{\delta ^{2}}\right )

    where :math:`\delta` (delta) is a hyperparameter, :math:`\bm{x}` and :math:`\bm{y}` are the
    input and output tensors, respectively.

    Args:
        delta (float, optional): Specify the Arctan cost. The value must be positive. Default: 1.0

    Note:
        The input has to be a non-negative tensor and the output tensor has the same shape with the
        input.

    Example:
        >>> import pypose.optim.kernel as ppok
        >>> kernel = ppok.Arctan()
        >>> input = torch.tensor([0, 0.5, 1, 2, 3])
        >>> kernel(input)
        tensor([0.0000, 0.4636, 0.7854, 1.1071, 1.2490])
    """
    def __init__(self, delta: float = 1.0) -> None:
        super().__init__()
        assert delta > 0, ValueError("delta has to be positive: {}".format(delta))
        self.delta1 = delta
        self.delta2 = delta**2

    def forward(self, input: Tensor) -> Tensor:
        '''
        Args:
            input (torch.Tensor): the input tensor (non-negative).
        '''
        assert torch.all(input >= 0), 'input has to be non-negative'
        return self.delta2*(input/self.delta2).arctan()


class Tolerant(nn.Module):
    r"""The robust Tolerant kernel cost function.

    .. math::
        \bm{y}_i = \delta ^{2}b\log \left ( 1+\frac{e^{\left ( \bm{x}_i-a)/b \right )}}{\delta ^{^{2}}} \right )
        - \delta ^{2}b\log \left ( 1+\frac{e^{-a/b}}{\delta ^{2}} \right )

    where :math:`\delta` (delta) is a hyperparameter, :math:`\bm{x}`,a,b and :math:`\bm{y}` are the
    input and output tensors, respectively.

    Args:
        delta (float, optional): Specify the Tolerant cost. The value must be positive. Default: 1.0

    Note:
        The input has to be a non-negative tensor and the output tensor has the same shape with the
        input. The input a,b have the same shape with the input

    Example:
        >>> import pypose.optim.kernel as ppok
        >>> kernel = ppok.Tolerant()
        >>> input = torch.tensor([0, 0.5, 1, 2, 3])
        >>> a = torch.tensor([0, 0.5, 1, 2, 3])
        >>> b = torch.tensor([0, 0.5, 1, 2, 3])
        >>> kernel(input, a, b)
        tensor([0.0000, 0.4636, 0.7854, 1.1071, 1.2490])
    """
    def __init__(self, delta: float = 1.0) -> None:
        super().__init__()
        assert delta > 0, ValueError("delta has to be positive: {}".format(delta))
        self.delta1 = delta
        self.delta2 = delta**2

    def forward(self, input: Tensor, a: float, b: float) -> Tensor:
        '''
        Args:
            input (torch.Tensor): the input tensor (non-negative).
        '''
        assert torch.all(input >= 0), 'input has to be non-negative'
        return self.delta2 * b * (1 + math.exp((input-a)/b)/self.delta2).log() - self.delta2 * b * (1 + math.exp(-a/b)/self.delta2)



class Scale(nn.Module):
    r"""The robust Scale kernel cost function.

    .. math::
        \bm{y}_i=a*\bm{x}_i

    where a is scale parameter, :math:`\bm{x}`, :math:`\bm{y}` is the input and output kernal function, respectively.


    Note:
        The input has to be a kernal function and the output  has the same shape with the
        input.

    Example:
        >>> import pypose.optim.kernel as ppok
        >>> kernel = ppok.Scale()
        >>> input = ppok.Huber
        >>> kernel(input, 2.0)
    """
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input:kernal , a: float):
        '''
        Args:
            input (optim.kernal): the input kernal.
        '''
        return a * input

