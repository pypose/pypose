import torch
import pytest
import pypose as pp


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('order', [[0, 1, 2, 3, 4], [1, 0, 2, 4, 3]])
def test_radius_filter_keeps_original_neighbor_indices(dtype, order):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    points = torch.tensor([[100., 11.], [0., 20.], [1., 30.],
                           [2.5, 50.], [50., 70.]], dtype=dtype, device=device)[order]
    original = points.clone()
    expected = torch.tensor([[0.5, 25.], [0.5, 25.], [1.75, 40.]],
                            dtype=dtype, device=device)

    actual = pp.knn_filter(points, k=1, pdim=1, radius=1.6)

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(points, original)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_radius_filter_can_use_neighbors_that_are_not_query_inliers(dtype):
    points = torch.tensor([[-0.9, 2.], [0., 4.], [1.1, 8.], [20., 32.]], dtype=dtype)
    # Only the middle point has two neighbors within the radius.
    expected = points[:3].mean(dim=0, keepdim=True)

    actual = pp.knn_filter(points, k=2, pdim=1, radius=1.2)

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_radius_filter_with_no_inliers(dtype):
    points = torch.tensor([[0., 1.], [2., 2.], [5., 3.]], dtype=dtype)
    actual = pp.knn_filter(points, k=1, pdim=1, radius=0.1)
    assert actual.shape == (0, 2)
    assert actual.dtype == dtype


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_filter_without_radius_preserves_batch_dimensions(dtype):
    points = torch.tensor([[0., 10.], [1., 20.], [3., 30.], [8., 40.]], dtype=dtype)
    expected = torch.tensor([[0.5, 15.], [0.5, 15.], [2., 25.], [5.5, 35.]], dtype=dtype)
    offsets = torch.arange(6, dtype=dtype).view(2, 3, 1, 1)

    actual = pp.knn_filter(points + offsets, k=1, pdim=1)

    torch.testing.assert_close(actual, expected + offsets)


def test_radius_filter_gradients_reach_original_neighbors():
    points = torch.tensor([[100., 11.], [0., 20.], [1., 30.],
                           [2.5, 50.], [50., 70.]], dtype=torch.float64,
                          requires_grad=True)

    pp.knn_filter(points, k=1, pdim=1, radius=1.6).sum().backward()

    weights = points.new_tensor([0., 1., 1.5, 0.5, 0.])
    torch.testing.assert_close(points.grad, weights[:, None].expand_as(points))
