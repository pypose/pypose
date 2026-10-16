from torch import Tensor
from torch.nn.modules.loss import _Loss
from .. import LieTensor, is_lietensor


def geodesic_loss(input:LieTensor, target:LieTensor, reduction:str = 'mean') -> Tensor:
    r'''Apply geodesic loss of the rotation part of input LieTensor and target LieTensor.
    See :class:`pypose.module.GeodesicLoss` for details.

    Examples:
        >>> input = pp.randn_SO3(2)
        SO3Type LieTensor:
        LieTensor([[ 0.0096, -0.0375,  0.0118,  0.9992],
                   [ 0.3858, -0.3133,  0.4539,  0.7396]])
        >>> target = pp.randn_SO3(2)
        SO3Type LieTensor:
        LieTensor([[ 0.3575,  0.0547, -0.6701,  0.6482],
                   [ 0.3086,  0.1362,  0.2598,  0.9048]])
        >>> geodesic_loss(input, target)
        tensor(1.4034)
    '''

    # input and target should be LieTensor
    assert is_lietensor(input) and is_lietensor(target), "input should be LieTensor"
    assert reduction in ['none', 'mean', 'sum'], "reduction type not supported"

    x, y = input.rotation(), target.rotation()
    e = x * y.Inv()
    if not e.ltype.on_manifold:
        e = e.Log()
    theta = e.norm(p='fro', dim=-1)

    if reduction == 'none':
        return theta # return batched tensors
    elif reduction == 'mean':
        return theta.mean() # return scalar
    elif reduction == 'sum':
        return theta.sum() # return scalar


class GeodesicLoss(_Loss):
    r"""Creates a criterion that measures the Geodesic Error between the rotation part of
    the input LieTensor :math:`x` and target LieTensor :math:`y`.

    .. math::
        \operatorname{norm}(\operatorname{Log}(\mathbf{x}\times\mathbf{y}^{-1})).

    Warning:
        The above equation holds only if :math:`x` and :math:`y` are quaternions.
        When :math:`x` and :math:`y` are rotation matrices, the geodesic error is

        .. math::
            \arccos \left(\frac{\operatorname{trace}(
            \mathbf{x}\times\mathbf{y}^{-1})-1}{2}\right).

        This class accepts all forms of input LieTensors, including ``SO3``, ``SE3``,
        ``RxSO3``, ``Sim3``, ``so3``, ``se3``, ``rxso3``, ``sim3``, where their rotation
        part will be extracted.

    Args:
        reduction (``str``, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction is applied,
            ``'mean'``: the sum of the output is divided by the number of
            elements in the output, ``'sum'``: the output is summed. Default: ``'mean'``

    Examples:
        >>> input, target = pp.randn_SO3(2), pp.randn_SO3(2)
        >>> criterion = pp.module.GeodesicLoss(reduction='mean')
        >>> loss = criterion(input, target)
        tensor(1.0112)
    """
    __constants__ = ["reduction"]

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__(size_average=None, reduce=None, reduction=reduction)

    def forward(self, input:LieTensor, target:LieTensor) -> Tensor:
        '''
        Args:
            input (``pp.LieTensor``): input LieTensor.
            target (``pp.LieTensor``): target LieTensor.
        '''
        return geodesic_loss(input, target, self.reduction)
