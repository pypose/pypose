import torch
from .ops import pm

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


def homo2cart(coordinates:torch.Tensor, eps=1e-12):
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
    denum = coordinates[...,-1:].abs().clamp_(min=eps)
    denum = pm(coordinates) * denum
    return coordinates[...,:-1] / denum
