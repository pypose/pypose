import math
import torch
from . import bvv
from .. import mat2SE3
from ..basics import pm
from .. import lietensor
from .. import LieTensor
from torch import broadcast_shapes


def is_lietensor(obj):
    r'''
    Check whether an instance or object is a LieTensor or not.

    Args:
        obj (``obj``): a Python object or instantance.

    Return:
        ``bool``: ``True`` if obj is a LieTensor object otherwise ``False``.
    '''
    return True if isinstance(obj, LieTensor) else False

def is_SE3(obj):
    r'''
    Check whether an instance or object is an SE3 Type LieTensor or not.

    Args:
        obj (``obj``): a Python object or instantance.

    Return:
        ``bool``: ``True`` if obj is a SE3 Type LieTensor object otherwise ``False``.
    '''
    return True if isinstance(obj.ltype, lietensor.lietensor.SE3Type) else False


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
    by dividing the last row. Size of the last dimension will be reduced by 1.

    Args:
        coordinates (``torch.Tensor``): the Homogeneous coordinates to be converted.

    Returns:
        ``torch.Tensor``: the coordinates in Cartesian space.

    Example:
        >>> points = torch.tensor([[4., 3., 2., 1.], [8., 6., 4., 2.]])
        >>> homo2cart(points)
        tensor([[4., 3., 2.],
                [4., 3., 2.]])
    '''
    tiny = torch.finfo(coordinates.dtype).tiny
    denum = coordinates[..., -1:].abs().clamp_(min=tiny)
    denum = pm(coordinates[..., -1:]) * denum
    return coordinates[...,:-1] / denum


def point2pixel(points, intrinsics, extrinsics=None):
    r'''
    Project batched sets of points (either in camera or world frame) to pixels.

    Args:
        points (``torch.Tensor``): The 3D coordinate of points. Assumed to be in the
            camera frame if ``extrinsics`` is ``None``, otherwiwse in the world frame.
            The shape has to be (..., N, 3).
        intrinsics (``torch.Tensor``): The intrinsic parameters of cameras.
            The shape has to be (..., 3, 3).
        extrinsics (``pypose.LieTensor``, optional): The extrinsic parameters of cameras.
            The shape has to be (..., 7). Default: ``None``.

    Returns:
        ``torch.Tensor``: The associated pixel with shape (..., N, 2).

    Example:
        >>> import torch, pypose as pp
        >>> f, (H, W) = 2, (9, 9) # focal length and image height, width
        >>> intrinsics = torch.tensor([[f, 0, H / 2],
        ...                            [0, f, W / 2],
        ...                            [0, 0,   1  ]])
        >>> object = torch.tensor([[2., 0., 2.],
        ...                        [1., 0., 2.],
        ...                        [0., 1., 1.],
        ...                        [0., 0., 1.],
        ...                        [1., 0., 1.],
        ...                        [5., 5., 3.]])
        >>> pixels = pp.point2pixel(object, intrinsics)
        tensor([[6.5000, 4.5000],
                [5.5000, 4.5000],
                [4.5000, 6.5000],
                [4.5000, 4.5000],
                [6.5000, 4.5000],
                [7.8333, 7.8333]])
        >>> pose = pp.SE3([ 0., -8,  0.,  0., -0.3827,  0.,  0.9239])
        >>> pixels = pp.point2pixel(object, intrinsics, pose)
        tensor([[  4.4999,  -1.1568],
                [  3.8332,  -3.0425],
                [  2.4998, -15.2997],
                [  2.4998, -18.1282],
                [  4.4999,  -6.8135],
                [  4.9999,   3.4394]])
    '''
    assert points.size(-1) == 3, "Points shape incorrect"
    assert intrinsics.size(-1) == intrinsics.size(-2) == 3, "Intrinsics shape incorrect."
    if extrinsics is None:
        broadcast_shapes(points.shape[:-2], intrinsics.shape[:-2])
    else:
        assert is_lietensor(extrinsics) and extrinsics.shape[-1] == 7, "Type incorrect."
        broadcast_shapes(points.shape[:-2], intrinsics.shape[:-2], extrinsics.shape[:-1])
        points = extrinsics.unsqueeze(-2) @ points
    return homo2cart(points @ intrinsics.mT)


def reprojerr(points, pixels, intrinsics, extrinsics=None):
    r'''
    Calculates batched per-pixel reprojection error (pixel distance) for points either in
    the camera or world frame given camera intrinsics or extrinsics, respectively.

    Args:
        points (``torch.Tensor``): The 3D coordinate of points. Assumed to be in the
            camera frame if ``extrinsics`` is ``None``, otherwiwse in the world frame.
            The shape has to be (..., N, 3).
        pixels (``torch.Tensor``): The image points. The associated pixel.
            The shape has to be (..., N, 2).
        intrinsics (``torch.Tensor``): intrinsic matrices.
            The shape has to be (..., 3, 3).
        extrinsics (``LieTensor``, optional): The camera extrinsics.
            The shape has to be (..., 7). Default: ``None``.
    Returns:
        Per-pixel reprojection error. The shape is (..., N).

    Example:
        >>> import torch, pypose as pp
        >>> f, (H, W) = 2, (9, 9) # focal length and image height, width
        >>> intrinsics = torch.tensor([[f, 0, H / 2],
        ...                            [0, f, W / 2],
        ...                            [0, 0,   1  ]])
        >>> object = torch.randn(6, 3)
        >>> pose = pp.randn_SE3()
        >>> pixels = pp.point2pixel(object, intrinsics, pose)
        >>> err = pp.reprojerr(object, pixels, intrinsics, pose)
        tensor([0., 0., 0., 0., 0., 0.])
    '''
    broadcast_shapes(points.shape[:-2], pixels.shape[:-2], intrinsics.shape[:-2])
    assert points.size(-1) == 3 and pixels.size(-1) == 2 and \
           intrinsics.size(-1) == intrinsics.size(-2) == 3, "Shape not compatible."
    img_repj = point2pixel(points, intrinsics, extrinsics)
    return (img_repj - pixels).norm(dim=-1)


def knn(pc1, pc2, ord=2, k=1, dim=-1, largest=False, sorted=True):
    r'''
    Select the k nearest neighbor points of pointcloud 1 from pointcloud 2 in each batch.

    Args:
        pc1 (``torch.Tensor``): The coordinates of the pointcloud 1.
            The shape has to be (..., N1, :).
        pc2 (``torch.Tensor``): The coordinates of the pointcloud 2.
            The shape has to be (..., N2, :).
        ord (``int``, optional): The order of norm to use for distance calculation.
            Default: ``2`` (Euclidean distance).
        k (``int``, optional): The number of the nearest neighbors to be selected.
            k has to be k :math:`\leq` N2. Default: ``1``.
        dim (``int``, optional): The dimension to sort along. Default: ``-1`` (The last
            dimension).
        largest (``bool``, optional): Return the k nearest or furthest neighbors. If
            ``largest`` is set to ``True``, then the k furthest neighbors are returned.
            Default: ``False``.
        sorted (``bool``, optional): Return the sorted or unsorted k nearest neighbors. If
            ``sorted`` is set to ``True``, it will make sure that the returned k nearest
            neighbors are themselves sorted. Default: ``True``.

    Note:
        If ``sorted`` is set to ``False``, the output will be unordered and not
        necessarily aligned with the index of the input point cloud.

    Returns:
        ``torch.return_types.topk (values: torch.Tensor, indices: torch.Tensor)``: The
        named tuple of (values, indices).

        ``values``: The ord-norm distance between each point in pc1 and its sorted k
        nearest neighbors in pc2. The shape is (..., N1, k).

        ``indices``: The index of the k nearest neighbor points in pc2.
        The shape is (..., N1, k).

    Example:
        >>> import torch, pypose as pp
        >>> pc1 = torch.tensor([[9., 2., 2.],
        ...                     [1., 0, 2.],
        ...                     [0., 1., 1.],
        ...                     [5., 0., 1.],
        ...                     [1., 0., 1.],
        ...                     [5., 5., 3.]])
        >>> pc2 = torch.tensor([[1., 0., 1.],
        ...                     [1., 6., 2.],
        ...                     [5., 1., 0.],
        ...                     [9., 0., 2.]])
        >>> pp.knn(pc1, pc2)
        torch.return_types.topk(
        values=tensor([[2.0000],
                [1.0000],
                [1.4142],
                [1.4142],
                [0.0000],
                [4.2426]]),
        indices=tensor([[3],
                [0],
                [0],
                [2],
                [0],
                [1]]))
        >>> pp.knn(pc1, pc2, k=2, ord=2)
        torch.return_types.topk(
        values=tensor([[2.0000, 4.5826],
                [1.0000, 4.5826],
                [1.4142, 5.0990],
                [1.4142, 4.0000],
                [0.0000, 4.2426],
                [4.2426, 5.0000]]),
        indices=tensor([[3, 2],
                [0, 2],
                [0, 2],
                [2, 0],
                [0, 2],
                [1, 2]]))
        >>> pp.knn(pc1, pc2, k=2, ord=2).values
        tensor([[2.0000, 4.5826],
                [1.0000, 4.5826],
                [1.4142, 5.0990],
                [1.4142, 4.0000],
                [0.0000, 4.2426],
                [4.2426, 5.0000]])
    '''
    diff = pc1.unsqueeze(-2) - pc2.unsqueeze(-3)
    dist = torch.linalg.norm(diff, dim=-1, ord=ord)
    neighbors = dist.topk(k, dim=dim, largest=largest, sorted=sorted)
    return neighbors


def svdtf(pc1, pc2):
    r'''
    Computes the rigid transformation ( :math:`SE(3)` ) between two sets of associated
    points using Singular Value Decomposition (SVD).

    Args:
        pc1 (``torch.Tensor``): The coordinates of the first set of points.
            The shape has to be (..., N1, 3).
        pc2 (``torch.Tensor``): The coordinates of the second set of points.
            The shape has to be (..., N2, 3).

    Returns:
        ``LieTensor``: The rigid transformation matrix in ``SE3Type``  that
        minimizes the mean squared error between the input point sets.

    Example:
        >>> import torch, pypose as pp
        >>> pc1 = torch.tensor([[0., 0., 0.],
        ...                     [1., 0., 0.],
        ...                     [0., 1., 0.]])
        >>> pc2 = torch.tensor([[1., 1., 1.],
        ...                     [2., 1., 1.],
        ...                     [1., 2., 1.]])
        >>> pp.svdtf(pc1, pc2)
        SE3Type LieTensor:
        LieTensor([1., 1., 1., 0., 0., 0., 1.])
    '''
    pc1ctn = pc1.mean(dim=-2, keepdim=True)
    pc2ctn = pc2.mean(dim=-2, keepdim=True)
    pc1t = pc1 - pc1ctn
    pc2t = pc2 - pc2ctn
    M = bvv(pc2t, pc1t).sum(dim=-3)
    U, S, Vh = torch.linalg.svd(M)
    R = U @ Vh
    mask = (R.det() + 1).abs() < 1e-6
    R[mask] = - R[mask]
    t = pc2ctn.mT - R @ pc1ctn.mT
    T = torch.cat((R, t), dim=-1)
    return mat2SE3(T, check=False)


def posediff(ref, est, aggregate=False, mode=1):
    r'''
    Computes the translatinal and rotational error between two batched transformations
    ( :math:`SE(3)` ).

    Args:
        ref (``LieTensor``): The reference transformation :math:`T_{ref}` in
            ``SE3type``. The shape is [..., 7].
        est (``LieTensor``): The estimated transformation :math:`T_{est}` in
            ``SE3type``. The shape is [..., 7].
        aggregate (``bool``, optional): Average the batched differences to a singleton
            dimension. Default: ``False``.
        mode (``int``, optional): Calculate the rotational difference in different mode.
            ``mode = 0``: Quaternions representation.
            ``mode = 1``: Axis-angle representation (Use one angle to represent the
            rotational difference in 3D space). Default: ``1``.

    Note:
        The rotation matrix to axis-angle representation refers to the theorem 2.5 and
        2.6 in Chapter 2 [1]. The implementation of the Quaternions to axis-angle
        representation (equation: :math:`\theta = 2 \cos^{-1}(q_0)` ) is presented at
        the end of Chapter 2 in [1].

        [1] Murray, R. M., Li, Z., & Sastry, S. S. (1994). A mathematical introduction to
        robotic manipulation. CRC press.


    Returns:
        ``torch.Tensor``: The translational difference (:math:`\Delta t`) and rotational
        differences between two sets of transformations.

        If ``aggregate = True``: The output batch will be 1.

        If ``mode = 0``: The values in each batch is :math:`[ \Delta t, \Delta q_x,
        \Delta q_y, \Delta q_z, \Delta q_w ]`

        If ``mode = 1``: The values in each batch is :math:`[ \Delta t, \Delta \theta ]`

    Example:
        >>> import torch, pypose as pp
        >>> ref = pp.randn_SE3(4)
        >>> est = pp.randn_SE3(4)
        >>> pp.posediff(ref,est)
        tensor([[3.1877, 0.3945],
        [3.3388, 2.0563],
        [2.4523, 0.4169],
        [1.8392, 1.1539]])
        >>> pp.posediff(ref,est,aggregate=True)
        tensor([1.9840, 1.9306])
        >>> pp.posediff(ref,est,mode=0)
        tensor([[ 3.1877,  0.1554,  0.1179, -0.0190,  0.9806],
        [ 3.3388, -0.0194, -0.8539,  0.0609,  0.5164],
        [ 2.4523,  0.0495, -0.1739,  0.1006,  0.9784],
        [ 1.8392, -0.5451, -0.0075,  0.0192,  0.8381]])
    '''
    assert is_SE3(ref), "The input reference transformation is not SE3Type."
    assert is_SE3(est), "The input estimated transformation is not SE3Type."
    assert mode in (0, 1), "Mode number is invalid."
    T = ref * est.Inv()
    diff_t = torch.linalg.norm(T.translation(), dim=-1, ord=2).unsqueeze(-1)
    if mode == 0:
        diff_r = T.rotation().tensor()
        diff = torch.cat((diff_t, diff_r), dim=-1)
    else:
        diff_r = 2 * torch.acos(T.tensor()[...,6])
        diff = torch.cat((diff_t, diff_r.unsqueeze(-1)), dim=-1)
    if aggregate and diff.ndim > 1:
        diff = diff.mean(dim=tuple(range(diff.ndim - 1)), keepdim=True).flatten()
    return diff
