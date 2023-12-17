import torch
from typing import Iterable
from torch.nn.modules.loss import _Loss
from torch.nn.functional import mse_loss
from .. import SO3, so3, LieTensor, SO3_type, so3_type

def geodesic_loss(
    input: LieTensor,
    target: LieTensor,
    reduction: str = "mean",
) -> torch.Tensor:
    r'''Apply geodesic loss to input rotation and target rotation:

    .. math::
        \arccos \left(\frac{\operatorname{trace}\left(\mathbf{x}_i \mathbf{x}_t^{-1}\right)-1}{2}\right).

    Args:
        input (``pp.LieTensor``): input rotation, should be pp.SO3 or pp.so3.
        target (``pp.LieTensor``): target rotation, should be pp.SO3 or pp.so3,
            target ltype should be the same as input's.
        reduction (``str``, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Default: ``'mean'``

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

    # input and target should be so3 or SO3
    if  input.ltype != so3_type and input.ltype != SO3_type:
        raise TypeError("input should be pp.SO3 or pp.so3")
    if target.ltype != so3_type and target.ltype != SO3_type:
        raise TypeError("target should be pp.SO3 or pp.so3")

    # input and target should have the same ltype
    if input.ltype != target.ltype:
        raise TypeError("input and target should have the same ltype")

    # if input.ltype is pp.so3, transfer to pp.SO3
    if input.ltype == so3_type:
        input = input.Exp()
        target = target.Exp()

    # caluculate geodesic loss
    rot_loss = (input * target.Inv()).Log()

    if reduction == "none":
        rot_loss = rot_loss
    elif reduction == "mean":
        rot_loss = rot_loss.norm(dim=-1).mean()
    elif reduction == "sum":
        rot_loss = rot_loss.norm(dim=-1).sum()
    else:
        rot_loss = input
        raise ValueError(reduction + " is not valid")

    return rot_loss

def pose_loss(
    input: LieTensor,
    target: LieTensor,
    weight: Iterable = [1., 1.],
    rot_loss_type: str = "geodesic",
    trans_loss_type: str = "mse",
    reduction: str = "mean",
) -> torch.Tensor:

    r'''Apply geodesic loss to input rotation and target rotation:

    .. math::
        \arccos \left(\frac{\operatorname{trace}\left(\mathbf{x}_i \mathbf{x}_t^{-1}\right)-1}{2}\right).

    Args:
        input (``pp.LieTensor``): input rotation, should be pp.SO3 or pp.so3.
        target (``pp.LieTensor``): target rotation, should be pp.SO3 or pp.so3,
            target ltype should be the same as input's.
        weight (``Iterable``, optional): weight for rotational loss and translational loss.
            Default: [1., 1.]
        rot_loss_type (``str``, optional): Specifies the rotational loss to apply to the output:
            Default: ``'geodesic'``
        trans_loss_type (``str``, optional): Specifies the translational loss to apply to the output:
            Default: ``'mse'``
        reduction (``str``, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Default: ``'mean'``

    Examples:
        >>> input = pp.randn_SE3(2)
        SE3Type LieTensor:
        LieTensor([[-1.5136, -0.9910, -1.3334, -0.0778, -0.1062, -0.2841,  0.9497],
                [-1.1133, -0.3473, -0.6473,  0.0043, -0.0361,  0.0190,  0.9992]])
        >>> target = pp.randn_SE3(2)
        SE3Type LieTensor:
        LieTensor([[ 0.5871,  0.7709, -0.1643, -0.0430,  0.1638, -0.4960,  0.8516],
                [ 1.3421,  0.4706,  1.4442,  0.3360, -0.0131, -0.2771,  0.9001]])
        >>> pose_loss(input, target)
        tensor(4.1468)
    '''

    # input and target should have the same ltype
    if  input.ltype != target.ltype:
        raise TypeError("input and target should have the same ltype")

    # calculate rotational loss
    input_rot = input.rotation()
    target_rot = target.rotation()

    if rot_loss_type == "geodesic":
        rot_loss = geodesic_loss(input_rot, target_rot, reduction=reduction)
    else:
        raise NotImplementedError(rot_loss_type + " is not implemented")

    # calculate translational loss
    input_trans = input.translation()
    target_trans = target.translation()

    if trans_loss_type == "mse":
        trans_loss = mse_loss(input_trans, target_trans, reduction=reduction)
    else:
        raise NotImplementedError(trans_loss_type + " is not implemented")

    if reduction == 'none':
        loss = torch.cat((rot_loss.tensor() * weight[0], trans_loss * weight[1]), dim=-1)
    else:
        loss = rot_loss * weight[0] + trans_loss * weight[1]

    return loss
