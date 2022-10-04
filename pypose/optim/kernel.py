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

    .. figure:: /_static/img/optim/kernel/huber.png
                        :width: 600
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

    .. figure:: /_static/img/optim/kernel/pseudohuber.png
                        :width: 600
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

    where :math:`\delta` (delta) is a hyperparameter, :math:`\bm{x}`
    and :math:`\bm{y}` are the input and output tensors, respectively.

    Args:
        delta (float, optional): Specify the Cauchy cost. The value must be positive. Default: 1.0

    Note:
        The input has to be a non-negative tensor and the output tensor
        has the same shape with the input.

    Example:
        >>> import pypose.optim.kernel as ppok
        >>> kernel = ppok.Cauchy()
        >>> input = torch.tensor([0, 0.5, 1, 2, 3])
        >>> kernel(input)
        tensor([0.0000, 0.4055, 0.6931, 1.0986, 1.3863])

    .. figure:: /_static/img/optim/kernel/cauchy.png
                        :width: 600
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
        \bm{y}_i=2\left ( \delta \sqrt{\frac{1}{{\delta{}}^{2}}+\bm{x}_i}- 1\right )

    where :math:`\delta` (delta) is a hyperparameter, :math:`\bm{x}` and :math:`\bm{y}`
    are the input and output tensors, respectively.

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

     .. figure:: /_static/img/optim/kernel/softlone.png
                        :width: 600
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
        return 2 * (self.delta1 * (1 / self.delta2 + input).sqrt() - 1)


class Arctan(nn.Module):
    r"""The robust Arctan kernel cost function.

    .. math::
        \bm{y}_i=\delta ^{2}\arctan \left ( \frac{\bm{x}_i}{\delta ^{2}}\right )

    where :math:`\delta` (delta) is a hyperparameter, :math:`\bm{x}` and :math:`\bm{y}` are the
    input and output tensors, respectively.

    Args:
        delta (float, optional): Specify the Arctan cost. Default: 1.0

    Note:
        The input has to be a non-negative tensor and the output tensor has the same shape with the
        input.

    Example:
        >>> import pypose.optim.kernel as ppok
        >>> kernel = ppok.Arctan()
        >>> input = torch.tensor([0, 0.5, 1, 2, 3])
        >>> kernel(input)
        tensor([0.0000, 0.4636, 0.7854, 1.1071, 1.2490])

    .. figure:: /_static/img/optim/kernel/arctan.png
                        :width: 600
    """
    def __init__(self, delta: float = 1.0) -> None:
        super().__init__()
        self.delta2 = delta**2

    def forward(self, input: Tensor) -> Tensor:
        '''
        Args:
            input (torch.Tensor): the input tensor (non-negative).
        '''
        assert torch.all(input >= 0), 'input has to be non-negative'
        return self.delta2 * (input / self.delta2).arctan()


class Tolerant(nn.Module):
    r"""The robust Tolerant kernel cost function.

    .. math::
        \bm{y}_i = b\log (1+e^{\frac{\bm{x}_i-a}{b}})-b\log (1+e^{\frac{-a}{b}})

    where :math:`\bm{a}`  and :math:`\bm{b}` are hyperparameters, :math:`\bm{x}`
    and :math:`\bm{y}` are the input and output tensors, respectively.

    Args:
        a (float): Specify the Tolerant cost. The value must be positive. Default: 1.0
        b (float): Specify the Tolerant cost. The value must be negative. Default: -1.0

    Note:
        The input has to be a non-negative tensor and the output tensor has the same shape with
        the input.

    Example:
        >>> import pypose.optim.kernel as ppok
        >>> kernel = ppok.Tolerant()
        >>> input = torch.tensor([0, 0.5, 1, 2, 3])
        >>> kernel(input)
        tensor([0.0000, 0.4636, 0.7854, 1.1071, 1.2490])

    .. figure:: /_static/img/optim/kernel/tolerant.png
                        :width: 600
    """
    def __init__(self, a: float = 1.0, b: float = -1.0) -> None:
        super().__init__()
        assert a > 0, ValueError("a has to be positive: {}".format(a))
        assert b < 0, ValueError("b has to be negative: {}".format(b))
        self.a = a
        self.b = b

    def forward(self, input: Tensor) -> Tensor:
        '''
        Args:
            input (torch.Tensor): the input tensor (non-negative).
        '''
        assert torch.all(input >= 0), 'input has to be non-negative'
        part1 = (1 + ((input-self.a) / self.b).exp()).log()
        part2 = (1 + (-self.a / self.b).exp())
        return self.b * part1 - self.b * part2


class Scale(nn.Module):
    r"""The robust Scale kernel cost function.

    .. math::
        \bm{y}_i=\delta*\bm{x}_i

    where :math:`\delta` (delta) is a scalar, :math:`\bm{x}` and :math:`\bm{y}` are the input
    and output tensors, respectively.

    Args:
        delta (float): Specify the scale factor. Should be between 0 and 1.  Default: 1.0

    Note:
        The input has to be a non-negative tensor and the output tensor has the same shape with
        the input.

    Example:
        >>> import pypose.optim.kernel as ppok
        >>> kernel = ppok.Scale()
        >>> input = torch.tensor([0, 0.5, 1, 2, 3])
        >>> kernel(input)
        tensor([0.0000, 0.5000, 1.0000, 2.0000, 3.0000])

     .. figure:: /_static/img/optim/kernel/scale.png
                        :width: 600
    """
    def __init__(self, delta: float = 1.0) -> None:
        super().__init__()
        assert 0 < delta <= 1 , ValueError("delta has to be between 0 and 1: {}".format(delta))
        self.delta = delta

    def forward(self, input):
        '''
        Args:
            input (torch.Tensor): the input tensor (non-negative).
        '''
        return self.delta * input

