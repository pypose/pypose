import torch
from .. import is_lietensor
from ..basics import homo2cart
from torch import broadcast_shapes


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
