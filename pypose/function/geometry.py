import torch
from ..basics import pm
from .. import is_lietensor
from torch import broadcast_shapes


def cart2homo(coordinates:torch.Tensor):
    r'''
    Converts batched Cartesian coordinates to Homogeneous coordinates
    by adding ones to last dimension.

    Args:
        coordinates (``torch.Tensor``): the Cartesian coordinates to be converted.

    Returns:
        ``torch.Tensor``: the coordinates in Homogeneous space.

    Note:
        The last dimension of the input coordinates can be any dimension.

    Example:

        >>> points = torch.randn(2, 2)
        >>> cart2homo(points)
        tensor([[ 2.0598,  1.5351,  1.0000],
                [-0.8484,  1.2390,  1.0000]])
        >>> points = torch.randn(2, 3)
        >>> cart2homo(points)
        tensor([[ 1.7946,  0.3548, -0.4446,  1.0000],
                [ 0.3010, -2.2748, -0.4708,  1.0000]])
    '''
    ones = torch.ones_like(coordinates[..., :1])
    return torch.cat([coordinates, ones], dim=-1)


def homo2cart(coordinates:torch.Tensor):
    r'''
    Converts batched Homogeneous coordinates to Cartesian coordinates
    by divising the last row.

    Args:
        coordinates (``torch.Tensor``): the Homogeneous coordinates to be converted.

    Returns:
        ``torch.Tensor``: the coordinates in Cartesian space.

    Example:
        >>> points = torch.randn(2, 4)
        tensor([[-0.5580, -1.1631, -0.5693,  0.5482],
                [-1.4770,  0.4703, -1.1718,  0.1307]])
        >>> homo2cart(points)
        tensor([[ -1.0179,  -2.1215,  -1.0385],
                [-11.2982,   3.5974,  -8.9636]])
    '''
    tiny = torch.finfo(coordinates.dtype).tiny
    denum = coordinates[..., -1:].abs().clamp_(min=tiny)
    denum = pm(coordinates[..., -1:]) * denum
    return coordinates[...,:-1] / denum


def point2pixel(points, intrinsics, extrinsics=None):
    r'''
    Project a set of points (either in camera or world frame) to pixels.

    Args:
        points (``torch.Tensor``): The object points in camera coordinate.
            The shape has to be (..., N, 3).
        intrinsics (``torch.Tensor``): The intrinsic matrices of cameras.
            The shape has to be (..., 3, 3).
        extrinsics (``pypose.LieTensor``, optional): The extrinsic transform of cameras.
            The shape has to be (..., 7). If ``None``, the points are assumed to be in
            the camera frame, otherwise in the world frame. Default: ``None``.

    Returns:
        ``torch.Tensor``: The associated pixel with shape (..., N, 2).
    '''
    assert points.size(-1) == 3, "Points shape incorrect"
    assert intrinsics.size(-1) == intrinsics.size(-2) == 3, "Intrinsics shape incorrect."
    if extrinsics is not None:
        assert is_lietensor(extrinsics) and extrinsics.shape[-1] == 7, "Type incorrect."
        points = extrinsics.unsqueeze(-2) @ points
    return homo2cart(points @ intrinsics.mT)


def reprojerr(points, pixels, intrinsics, extrinsics):
    r'''
    Performs batched re-projection and returns the per-pixel error (distance).

    Calculate the reprojection error for points in the world coordinate and their
    associated pixels, given camera pose and intrinsic matrices.

    Args:
        points (``torch.Tensor``): The object points in world coordinate.
            The shape has to be (..., N, 3).
        pixels (``torch.Tensor``): The image points. The associated pixel.
            The shape has to be (..., N, 2).
        intrinsics (``torch.Tensor``): intrinsic matrices.
            The shape has to be (..., 3, 3).
        extrinsics (``LieTensor``): The camera extrinsics.
            The shape has to be (..., 7).
    Returns:
        Per-pixel reprojection error. The shape is (..., N).
    '''
    broadcast_shapes(points.shape[:-2], pixels.shape[:-2], \
                     extrinsics.shape[:-1], intrinsics.shape[:-2])
    assert points.size(-1) == 3 and pixels.size(-1) == 2 and is_lietensor(extrinsics) \
        and intrinsics.size(-1) == intrinsics.size(-2) == 3, "Shape not compatible."
    img_repj = point2pixel(points, intrinsics, extrinsics)
    return (img_repj - pixels).norm(dim=-1)
