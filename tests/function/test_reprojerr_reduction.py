import torch
import pytest
import pypose as pp


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_reprojection_sum_is_l1_distance(dtype):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    points = torch.tensor([[2., 4., 2.], [6., 3., 3.], [-2., 6., 2.]],
                          dtype=dtype, device=device)
    intrinsics = torch.tensor([[2., 0., 5.], [0., 3., 7.], [0., 0., 1.]],
                              dtype=dtype, device=device)
    pixels = torch.tensor([[8., 12.], [11., 13.], [3., 16.]],
                          dtype=dtype, device=device)
    expected = torch.tensor([2., 5., 0.], dtype=dtype, device=device)

    actual = pp.reprojerr(points, pixels, intrinsics, reduction='sum')

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize('reduction', ['none', 'norm'])
def test_other_reprojection_reductions(reduction):
    points = torch.tensor([[1., 1., 1.], [2., 3., 1.]], dtype=torch.float64)
    pixels = torch.tensor([[2., 0.], [4., 6.]], dtype=torch.float64)
    expected = torch.tensor([[-1., 1.], [-2., -3.]], dtype=torch.float64)
    if reduction == 'norm':
        expected = torch.tensor([2., 13.], dtype=torch.float64).sqrt()

    actual = pp.reprojerr(points, pixels, torch.eye(3, dtype=torch.float64),
                          reduction=reduction)

    torch.testing.assert_close(actual, expected)


def test_reprojection_sum_with_broadcast_cameras_and_extrinsics():
    points = torch.tensor([[1., 2., 1.], [3., 4., 1.], [5., 6., 1.]], dtype=torch.float64)
    intrinsics = torch.eye(3, dtype=torch.float64).expand(2, 3, 3)
    extrinsics = pp.SE3(torch.tensor([[0., 0., 0., 0., 0., 0., 1.],
                                      [2., -1., 0., 0., 0., 0., 1.]],
                                     dtype=torch.float64))
    pixels = torch.tensor([[2., 1.], [4., 3.], [6., 5.]], dtype=torch.float64)

    actual = pp.reprojerr(points, pixels, intrinsics, extrinsics, reduction='sum')

    expected = torch.tensor([[2., 2., 2.], [1., 1., 1.]], dtype=torch.float64)
    torch.testing.assert_close(actual, expected)


def test_reprojection_sum_with_empty_point_set():
    points = torch.empty(2, 0, 3, dtype=torch.float64)
    pixels = torch.empty(0, 2, dtype=torch.float64)
    actual = pp.reprojerr(points, pixels, torch.eye(3, dtype=torch.float64),
                          reduction='sum')
    assert actual.shape == (2, 0)


def test_reprojection_sum_gradients():
    points = torch.tensor([[1., 1., 2.], [2., 3., 3.]], dtype=torch.float64,
                          requires_grad=True)
    pixels = torch.tensor([[2., 0.], [4., 6.]], dtype=torch.float64, requires_grad=True)
    intrinsics = torch.eye(3, dtype=torch.float64)
    actual = pp.reprojerr(points, pixels, intrinsics, reduction='sum')
    pixel_grad, = torch.autograd.grad(actual.sum(), pixels)
    torch.testing.assert_close(pixel_grad, pixels.new_tensor([[1., -1.], [1., 1.]]))
    assert torch.autograd.gradcheck(
        lambda p, q: pp.reprojerr(p, q, intrinsics, reduction='sum'), (points, pixels))
