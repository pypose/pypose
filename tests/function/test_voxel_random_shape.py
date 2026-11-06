import torch
import pytest
import pypose as pp


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('count', [1, 3])
def test_random_voxel_filter_preserves_single_voxel_axis(dtype, count):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    points = torch.tensor([[1., 2., 7.], [1.2, 2.4, 8.], [1.5, 2.2, 9.]],
                          dtype=dtype, device=device)[:count]
    original = points.clone()

    actual = pp.voxel_filter(points, [2., 2.], random=True)

    assert actual.shape == (1, 3)
    assert actual.dtype == dtype
    assert actual.device == points.device
    assert torch.any(torch.all(actual[0] == points, dim=-1))
    torch.testing.assert_close(points, original)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_random_voxel_filter_selects_one_complete_point_per_voxel(dtype):
    points = torch.tensor([[0., 0., 10.], [0.5, 0., 20.], [2.2, 0., 30.],
                           [2.5, 0., 40.], [4., 0., 50.]], dtype=dtype)
    points = points.t().contiguous().t()
    assert not points.is_contiguous()

    actual = pp.voxel_filter(points, [2., 2.], random=True)

    assert actual.shape == (3, 3)
    for selected, group in zip(actual, [points[:2], points[2:4], points[4:]]):
        assert torch.any(torch.all(selected == group, dim=-1))


@pytest.mark.parametrize('count', [1, 3])
def test_centroid_voxel_filter_retains_single_voxel_shape(count):
    points = torch.tensor([[1., 2., 7.], [1.2, 2.4, 8.], [1.5, 2.2, 9.]],
                          dtype=torch.float64)[:count]
    actual = pp.voxel_filter(points, [2., 2.])
    torch.testing.assert_close(actual, points.mean(dim=0, keepdim=True))


def test_random_single_point_voxel_retains_gradients():
    points = torch.tensor([[1., 2., 7.]], dtype=torch.float64, requires_grad=True)
    actual = pp.voxel_filter(points, [2., 2.], random=True)
    assert actual.shape == points.shape

    actual.sum().backward()

    torch.testing.assert_close(points.grad, torch.ones_like(points))


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('random', [False, True])
def test_empty_voxel_filter_preserves_shape_and_gradients(dtype, random):
    points = torch.empty(0, 3, dtype=dtype, requires_grad=True)

    actual = pp.voxel_filter(points, [2., 2.], random=random)

    assert actual.shape == points.shape
    assert actual.dtype == points.dtype
    assert actual.device == points.device
    actual.sum().backward()
    torch.testing.assert_close(points.grad, torch.empty_like(points))
