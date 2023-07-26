import torch
from .. import mat2SE3
from ..basics import pm
from .checking import is_lietensor


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
        torch.broadcast_shapes(points.shape[:-2], intrinsics.shape[:-2])
    else:
        assert is_lietensor(extrinsics) and extrinsics.shape[-1] == 7, "Type incorrect."
        torch.broadcast_shapes(points.shape[:-2], intrinsics.shape[:-2], extrinsics.shape[:-1])
        points = extrinsics.unsqueeze(-2) @ points
    return homo2cart(points @ intrinsics.mT)


def reprojerr(points, pixels, intrinsics, extrinsics=None, reduction='none'):
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
        reduction (``str``, optional): The reduction to apply on the output: ``'none'``
            | ``'sum'`` | ``'norm'``

            ``'none'``: No reduction is applied

            ``'sum'``: The reprojection error on each component (u, v) is summed for
            each pixel (L1 Norm)

            ``'norm'``: The reprojection error's L2 norm for each pixel
    Returns:
        Per-pixel reprojection error.

        The shape is (..., N) if reduction is ``'sum'`` or ``'norm'``.

        The shape is (..., N, 2) if reduction is ``'none'``.

    Example:
        >>> import torch, pypose as pp
        >>> f, (H, W) = 2, (9, 9) # focal length and image height, width
        >>> intrinsics = torch.tensor([[f, 0, H / 2],
        ...                            [0, f, W / 2],
        ...                            [0, 0,   1  ]])
        >>> object = torch.randn(6, 3)
        >>> pose = pp.randn_SE3()
        >>> pixels = pp.point2pixel(object, intrinsics, pose)
        >>> err = pp.reprojerr(object, pixels, intrinsics, pose, reduction='norm')
        tensor([0., 0., 0., 0., 0., 0.])
    '''
    torch.broadcast_shapes(points.shape[:-2], pixels.shape[:-2], intrinsics.shape[:-2])
    assert points.size(-1) == 3 and pixels.size(-1) == 2 and \
           intrinsics.size(-1) == intrinsics.size(-2) == 3, "Shape not compatible."
    assert reduction in {'norm', 'sum', 'none'}, \
           "Reduction method can only be 'norm'|'sum'|'none'."
    img_repj = point2pixel(points, intrinsics, extrinsics)

    if reduction == 'norm':
        return (img_repj - pixels).norm(dim=-1)
    elif reduction == 'sum':
        return (img_repj - pixels).sum(dim=-1)
    return img_repj - pixels


def knn(ref, nbr, k=1, ord=2, dim=-1, largest=False, sorted=True):
    r'''
    Select the k nearest neighbor points of reference from neighbors in each batch.

    Args:
        ref (``torch.Tensor``): the coordinates of the reference point sets.
            The shape has to be (..., N1, :).
        nbr (``torch.Tensor``): the coordinates of the neighbors point sets.
            The shape has to be (..., N2, :).
        k (``int``, optional): the number of the nearest neighbors to be selected.
            k has to be k :math:`\leq` N2. Default: ``1``.
        ord (``int``, optional): the order of norm to use for distance calculation.
            Default: ``2`` (Euclidean distance).
        dim (``int``, optional): the dimension encompassing the point cloud coordinates,
            utilized for calculating distance and sorting.
            Default: ``-1`` (The last dimension).
        largest (``bool``, optional): controls whether to return largest (furthest) or
            smallest (nearest) neighbors. Default: ``False``.
        sorted (``bool``, optional): controls whether to return the neighbors in sorted
            order. Default: ``True``.

    Returns:
        ``torch.return_types.topk(values: torch.Tensor, indices: torch.LongTensor)``:
        The named tuple of (values, indices).

        ``values``: The ord-norm distance between each point in ref and its sorted k
        nearest neighbors in nbr. The shape is (..., N1, k).

        ``indices``: The index of the k nearest neighbor points in neighbors point sets
        (nbr). The shape is (..., N1, k).

    Note:
        If ``sorted`` is set to ``False``, the output will be unspecified and not
        necessarily sorted along the index of the input point cloud.

    Example:
        >>> import torch, pypose as pp
        >>> ref = torch.tensor([[9., 2., 2.],
        ...                     [1., 0., 2.],
        ...                     [0., 1., 1.],
        ...                     [5., 0., 1.],
        ...                     [1., 0., 1.],
        ...                     [5., 5., 3.]])
        >>> nbr = torch.tensor([[1., 0., 1.],
        ...                     [1., 6., 2.],
        ...                     [5., 1., 0.],
        ...                     [9., 0., 2.]])
        >>> pp.knn(ref, nbr)
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
        >>> pp.knn(ref, nbr, k=2, ord=2)
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
        >>> pp.knn(ref, nbr, k=2, ord=2).values
        tensor([[2.0000, 4.5826],
                [1.0000, 4.5826],
                [1.4142, 5.0990],
                [1.4142, 4.0000],
                [0.0000, 4.2426],
                [4.2426, 5.0000]])
    '''
    diff = ref.unsqueeze(-2) - nbr.unsqueeze(-3)
    dist = torch.linalg.norm(diff, dim=dim, ord=ord)
    return dist.topk(k, dim=dim, largest=largest, sorted=sorted)


def svdtf(source, target):
    r'''
    Computes the rigid transformation ( :math:`SE(3)` ) between two sets of associated
    point clouds (source and target) using Singular Value Decomposition (SVD).

    Args:
        source (``torch.Tensor``): the coordinates of the source point cloud.
            The shape has to be (..., N, 3).
        target (``torch.Tensor``): the coordinates of the target point cloud.
            The shape has to be (..., N, 3).

    Returns:
        ``LieTensor``: The rigid transformation matrix in ``SE3Type``  that
        minimizes the mean squared error between the input point sets.

    Warning:
        The number of points N has to be the same for both point clouds.

    Example:
        >>> import torch, pypose as pp
        >>> source = torch.tensor([[0., 0., 0.],
        ...                     [1., 0., 0.],
        ...                     [0., 1., 0.]])
        >>> target = torch.tensor([[1., 1., 1.],
        ...                     [2., 1., 1.],
        ...                     [1., 2., 1.]])
        >>> pp.svdtf(source, target)
        SE3Type LieTensor:
        LieTensor([1., 1., 1., 0., 0., 0., 1.])
    '''
    assert source.size(-2) == target.size(-2), {
        "The number of points N has to be the same for both point clouds."}
    ctnsource = source.mean(dim=-2, keepdim=True)
    ctntarget = target.mean(dim=-2, keepdim=True)
    source = source - ctnsource
    target = target - ctntarget
    M = torch.einsum('...Na, ...Nb -> ...ab', target, source)
    U, S, Vh = torch.linalg.svd(M)
    R = U @ Vh
    mask = (R.det() + 1).abs() < 1e-6
    R[mask] = - R[mask]
    t = ctntarget.mT - R @ ctnsource.mT
    T = torch.cat((R, t), dim=-1)
    return mat2SE3(T, check=False)
