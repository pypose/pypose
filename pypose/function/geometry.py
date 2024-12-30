import torch
from typing import List
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


def pixel2point(pixels, depth, intrinsics):
    r'''
    Convert batch of pixels with depth into points (in camera coordinate)

    Args:
        pixels: (``torch.Tensor``) The 2d coordinates of pixels in the camera
            pixel coordinate.
            Shape has to be (..., N, 2)

        depth: (``torch.Tensor``) The depths of pixels with respect to the
            sensor plane.
            Shape has to be (..., N)

        intrinsics: (``torch.Tensor``): The intrinsic parameters of cameras.
            The shape has to be (..., 3, 3).

    Returns:
        ``torch.Tensor`` The associated 3D-points with shape (..., N, 3)

    Example:
        >>> import torch, pypose as pp
        >>> f, (H, W) = 2, (9, 9) # focal length and image height, width
        >>> intrinsics = torch.tensor([[f, 0, H / 2],
        ...                            [0, f, W / 2],
        ...                            [0, 0,   1  ]])
        >>> pixels = torch.tensor([[0.5, 0.0],
        ...                        [1.0, 0.0],
        ...                        [0.0, 1.3],
        ...                        [1.0, 0.0],
        ...                        [0.5, 1.5],
        ...                        [5.0, 1.5]])
        >>> depths = torch.tensor([5.0, 3.0, 6.5, 2.0, 0.5, 0.7])
        >>> points = pp.pixel2point(pixels, depths, intrinsics)
        tensor([[-10.0000, -11.2500,   5.0000],
                [ -5.2500,  -6.7500,   3.0000],
                [-14.6250, -10.4000,   6.5000],
                [ -3.5000,  -4.5000,   2.0000],
                [ -1.0000,  -0.7500,   0.5000],
                [  0.1750,  -1.0500,   0.7000]])
    '''
    assert pixels.size(-1) == 2, "Pixels shape incorrect"
    assert depth.size(-1) == pixels.size(-2), "Depth shape does not match pixels"
    assert intrinsics.size(-1) == intrinsics.size(-2) == 3, "Intrinsics shape incorrect."

    fx, fy = intrinsics[..., 0, 0], intrinsics[..., 1, 1]
    cx, cy = intrinsics[..., 0, 2], intrinsics[..., 1, 2]

    assert not torch.any(fx == 0), "fx Cannot contain zero"
    assert not torch.any(fy == 0), "fy Cannot contain zero"

    pts3d_z = depth
    pts3d_x = ((pixels[..., 0] - cx) * pts3d_z) / fx
    pts3d_y = ((pixels[..., 1] - cy) * pts3d_z) / fy
    return torch.stack([pts3d_x, pts3d_y, pts3d_z], dim=-1)


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


def nbr_filter(points, nbr:int, radius:float, pdim:int=None, ord:int=2, return_mask:bool=False):
    r'''
    Filter point outliers by checking if a point has less than n neighbors (nbr) within a
    radius.

    Args:
        points (``torch.Tensor``): the input point cloud. It is possible that the last
            dimension (D) is larger than ``pdim``, with point's coordinates using first
            ``pdim`` values. Subsequent values may contain additional information like
            intensity, RGB channels, etc. The shape has to be (N, D).
        nbr (``int``): the minimum number of neighbors (nbr) within a certain radius.
        ord (``int``, optional): the order of norm to use for distance calculation.
            Default: ``2`` (Euclidean distance).
        radius (``float``): the radius of the sphere for counting the neighbors.
        pdim (``int``, optional): the dimsion of points, where :math:`\text{pdim} \le D`.
            Default to the last dimension of points, if ``None``.
        return_mask (``bool``, optional): return the mask of inliers of not.

    Returns:
        ``torch.Tensor``: The point clouds removed outliers.
        ``torch.BoolTensor``: The mask of point clouds removed outliers, where the inlier
        is True and the outlier is False. The shape is (..., N).

    Warning:
        Note that this operation does not support batch operations, since the number of
        output voxels can be different on different batches.

    Example:
        >>> import torch, pypose as pp
        >>> points = torch.tensor([[0., 0., 0.],
        ...                        [1., 0., 0.],
        ...                        [0., 1., 0.],
        ...                        [0., 1., 1.],
        ...                        [10., 1., 1.],
        ...                        [10., 1., 10.]])
        >>> pp.nbr_filter(points, nbr=2, radius=5)
        tensor([[0., 0., 0.],
                [1., 0., 0.],
                [0., 1., 0.],
                [0., 1., 1.]])
        >>> pp.nbr_filter(points, nbr=2, radius=12, return_mask=True)
        (tensor([[ 0.,  0.,  0.],
                 [ 1.,  0.,  0.],
                 [ 0.,  1.,  0.],
                 [ 0.,  1.,  1.],
                 [10.,  1.,  1.]]),
        tensor([ True,  True,  True,  True,  True, False]))
    '''
    assert len(points.shape) == 2, "The point cloud dimension has to be 2."
    pdim = points.size(-1) if pdim == None else pdim
    assert points.size(-1) >= pdim, "The last dim of points should not less than pdim."
    diff = points[..., :pdim].unsqueeze(-2) - points[..., :pdim].unsqueeze(-3)
    count = torch.sum(torch.linalg.norm(diff, dim=-1, ord=ord) <= radius, dim=-1) - 1
    mask = count >= nbr
    if return_mask:
        return points[mask], mask
    else:
        return points[mask]


def random_filter(points:torch.Tensor, num:int):
    r'''
    Randomly sample a number of points from a batched point cloud.

    Args:
        points (``torch.Tensor``): the input point cloud, where the last dimension (D)
            is the dimension of the points. The shape should be (..., N, D), where N is
            the number of points.
        num (``int``): the number of points to sample.

    Returns:
        output (``torch.Tensor``): The sampled points, with the shape (..., num, D).

    Example:
        >>> import torch, pypose as pp
        >>> points = torch.tensor([[1., 2., 3.],
        ...                        [4., 5., 6.],
        ...                        [7., 8., 9.],
        ...                        [10., 11., 12.],
        ...                        [13., 14., 15.]])
        >>> pp.random_filter(points, 3)
        tensor([[ 4.,  5.,  6.],
                [ 1.,  2.,  3.],
                [10., 11., 12.]])
    '''
    assert points.size(-1) >= 1, "The last dim of the points should not less than 1."
    assert num <= points.size(-2), "Number of points to sample must not larger than " \
        "the number of input points."

    indices = torch.randperm(points.size(-2))[:num]
    # return torch.index_select(points, 0, indices)
    return points[..., indices, :]


def voxel_filter(points: torch.Tensor, voxel: List[float], random:bool = False):
    r'''
    Perform voxel filtering on a point cloud to reduce the number of points by grouping
    them into voxels and selecting a representative point for each voxel.

    Args:
        points (``torch.Tensor``): The input point cloud. It is possible that the last
            dimension (D) is larger than dimension of voxel :math:`v` (vdim), with the
            point's coordinates as the first :math:`v` values.
            Subsequent values are additional information such as intensity, RGB channels.
            The shape has to be (N, D), where :math:`D \geq v`.

        voxel (list of ``float``): The sizes of the voxel in each dimension, provided as
            :math:`\left[v_1, v_2, \cdots, v_{\text{dim}}\right]`.
        random (``bool``, optional): If ``True``, a random point within each voxel is
            chosen as the representative, othewise the centroid of the points is used.
            Default: ``False``.

    Returns:
        output (``torch.Tensor``): The sampled point cloud, with each point representing a
            voxel. The shape is (M, D), where M (:math:`M\le N`) is the number of voxels.

    Warning:
        Note that this operation does not support batch operations, since the number of
        output voxels can be differeent on each batch.

    Example:
        >>> import torch, pypose as pp
        >>> points = torch.tensor([[ 1.,  2.,  3.],
        ...                        [ 4.,  5.,  6.],
        ...                        [ 7.,  8.,  9.],
        ...                        [10., 11., 12.],
        ...                        [13., 14., 15.]])
        >>> pp.voxel_filter(points, [5., 5., 5.])
        tensor([[ 2.5000,  3.5000,  4.5000],
                [ 8.5000,  9.5000, 10.5000],
                [13.0000, 14.0000, 15.0000]])
        >>> pp.voxel_filter(points, [5., 5., 5.], random=True)
        tensor([[ 4.,  5.,  6.],
                [10., 11., 12.],
                [13., 14., 15.]])
    '''
    assert len(points.shape) == 2, "The point cloud dimension has to be 2."
    D, vdim = points.size(-1), len(voxel)
    assert D >= vdim, "The last dimension of the pointcloud should exceed \
        the dimenson of the voxel space."
    assert all(item != 0 for item in voxel), "Voxel size should be nonzero."
    kwargs = {'device': points.device, 'dtype': points.dtype}

    minp = torch.min(points[..., :vdim], dim=-2).values
    indices = ((points[..., :vdim] - minp) / torch.tensor(voxel)).to(torch.int64)

    unique_indices, inverse_indices, counts = torch.unique(
        indices, dim=-2, return_inverse=True, return_counts=True)
    if random:
        sorting_indices = torch.argsort(inverse_indices).squeeze()
        sorted_points = points[sorting_indices, :]
        _rand = [torch.randint(low=0, high=count.item(), size=(1,)) for count in counts]
        random_indices = torch.cat(_rand)
        selected_indices = (random_indices + torch.cumsum(counts, dim=0) - counts).squeeze()
        return sorted_points[..., selected_indices, :]
    else:
        means = torch.zeros_like(unique_indices, **kwargs)
        values = torch.zeros([len(unique_indices), D-vdim], **kwargs)
        voxels = torch.cat([means, values], dim=-1)
        counts = torch.zeros(unique_indices.size(0), **kwargs)

        voxels.index_add_(0, inverse_indices, points)
        _ones = torch.ones_like(inverse_indices, **kwargs)
        counts.index_add_(0, inverse_indices, _ones)
        voxels /= counts.view(-1, 1)

        return voxels


def knn_filter(points:torch.Tensor, k:int, pdim:int=None, radius:float=None, ord:int=2):
    r'''
    Filter batched points by averaging its k-nearest neighbors and that point, removing
    points if number of neighbors within radius is less than k.

    Args:
        points (``torch.Tensor``): the input point cloud, where the last dimension (D)
            is the dimension of the points. The shape should be (..., N, D), where N is
            the number of points in each batch.
        k (``int``): the number of neighbors within a radius.
        pdim (``int``, optional): the dimsion of points, where :math:`\text{pdim} \le D`.
            Default to the last dimension of points, if ``None``.
        radius (``float``, optional): the radius of the sphere for counting the neighbors.
            Not use if None,
            Default: ``None``
        ord (``int``, optional): the order of norm to use for distance calculation.
            Default: ``2`` (Euclidean distance).

    Returns:
        output (``torch.Tensor``): The sampled points, with the shape (..., num, D).

    Warning:
        This operation **supports** batch operations if ``radius`` is **not** given, where
        the dimension of input points is (..., N, D), otherwise it has to be (N, D).
        This is because given a radius to remove outliers, the number of output points in
        each batch can be different.

    Example:
        >>> import torch, pypose as pp
        >>> points = torch.tensor([[0.,  0.,  0.],
        ...                        [1.,  0.,  0.],
        ...                        [0.,  1.,  0.],
        ...                        [0.,  1.,  1.],
        ...                        [10., 1.,  1.],
        ...                        [10., 1., 10.]])
        >>> pp.knn_filter(points, k=2, radius=5)
        tensor([[0.3333, 0.3333, 0.0000],
                [0.3333, 0.3333, 0.0000],
                [0.0000, 0.6667, 0.3333],
                [0.0000, 0.6667, 0.3333]])
    '''
    if radius is not None:
        assert len(points.shape) == 2, "The points dimension has to be 2 given radius."
    else:
        assert len(points.shape) >= 2, "The points dimension cannot be less than 2."
    pdim = points.size(-1) if pdim == None else pdim
    assert points.size(-1) >= pdim, "The last dim of points should not less than pdim."
    diff = points[..., :pdim].unsqueeze(-2) - points[..., :pdim].unsqueeze(-3)
    dist = torch.linalg.norm(diff, dim=-1, ord=ord)

    if radius is not None:
        count = torch.sum(dist <= radius, dim=-1) - 1
        rmask = count >= k
        points, dist = points[rmask], dist[rmask]

    _, idx = dist.topk(k+1, dim=-1, largest=False, sorted=True)
    shape = points.size() + torch.Size([k+1])
    idx = idx.unsqueeze(-2).expand(shape) # expand to [B, D, K+1]
    points = points.unsqueeze(-1).expand(shape) # expand to [B, D, K+1]
    return torch.gather(points, -3, idx).mean(dim=-1)
