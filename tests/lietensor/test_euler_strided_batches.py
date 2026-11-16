import numpy as np
import pytest
import torch
import pypose as pp


def rotation_matrix(angles):
    roll, pitch, yaw = angles.unbind(-1)
    zero, one = torch.zeros_like(roll), torch.ones_like(roll)
    cr, sr, cp, sp, cy, sy = roll.cos(), roll.sin(), pitch.cos(), pitch.sin(), yaw.cos(), yaw.sin()
    shape = angles.shape[:-1] + (3, 3)
    rx = torch.stack((one, zero, zero, zero, cr, -sr, zero, sr, cr), -1).reshape(shape)
    ry = torch.stack((cp, zero, sp, zero, one, zero, -sp, zero, cp), -1).reshape(shape)
    rz = torch.stack((cy, -sy, zero, sy, cy, zero, zero, zero, one), -1).reshape(shape)
    return rz @ ry @ rx


def make_view(layout, dtype, requires_grad=False):
    generator = torch.Generator().manual_seed(504)
    if layout == 'transpose':
        raw = torch.randn(2, 4, 3, generator=generator, dtype=dtype, requires_grad=requires_grad)
        angles = raw.transpose(0, 1)
    elif layout == 'permute':
        raw = torch.randn(2, 3, 4, 3, generator=generator, dtype=dtype, requires_grad=requires_grad)
        angles = raw.permute(2, 0, 1, 3)
    elif layout == 'slice':
        raw = torch.randn(4, 6, 6, generator=generator, dtype=dtype, requires_grad=requires_grad)
        angles = raw[::2, ::2, ::2]
    elif layout == 'expand':
        raw = torch.randn(1, 4, 3, generator=generator, dtype=dtype, requires_grad=requires_grad)
        angles = raw.expand(3, 4, 3)
    elif layout == 'broadcast':
        raw = torch.randn(2, 1, 4, 3, generator=generator, dtype=dtype, requires_grad=requires_grad)
        angles = raw.expand(2, 3, 4, 3)
    else:
        raw = torch.randn(2, 3, 4, generator=generator, dtype=dtype, requires_grad=requires_grad)
        angles = raw.transpose(-1, -2)
    return raw, angles


LAYOUTS = ('transpose', 'permute', 'slice', 'expand', 'broadcast', 'coordinates')


@pytest.mark.parametrize('layout', LAYOUTS)
@pytest.mark.parametrize('dtype', (torch.float32, torch.float64))
def test_strided_batches_match_independent_axis_rotations(layout, dtype):
    raw, angles = make_view(layout, dtype)
    original = raw.clone()
    actual = pp.euler2SO3(angles)
    assert actual.ltype is pp.SO3_type
    assert actual.shape == angles.shape[:-1] + (4,)
    assert actual.dtype is dtype
    torch.testing.assert_close(actual.matrix(), rotation_matrix(angles))
    torch.testing.assert_close(actual.tensor(), pp.euler2SO3(angles.contiguous()).tensor(), atol=0, rtol=0)
    torch.testing.assert_close(raw, original)


@pytest.mark.parametrize('layout', LAYOUTS)
def test_view_gradients_match_existing_contiguous_conversion(layout):
    raw, angles = make_view(layout, torch.float64, requires_grad=True)
    coefficients = torch.tensor([0.2, -0.4, 0.7, 0.3], dtype=torch.float64)
    actual = (pp.euler2SO3(angles).tensor() * coefficients).sum()
    expected = (pp.euler2SO3(angles.contiguous()).tensor() * coefficients).sum()
    actual_gradient = torch.autograd.grad(actual, raw)[0]
    expected_gradient = torch.autograd.grad(expected, raw)[0]
    assert actual_gradient.abs().max() > 0.1
    torch.testing.assert_close(actual_gradient, expected_gradient, atol=1e-14, rtol=1e-14)


@pytest.mark.parametrize('shape', ((3,), (1, 3), (2, 4, 3), (0, 3), (2, 0, 3), (0, 2, 3)))
@pytest.mark.parametrize('dtype', (torch.float32, torch.float64))
def test_contiguous_and_empty_batches_keep_shape_and_rotation(shape, dtype):
    angles = torch.linspace(-0.8, 0.9, torch.Size(shape).numel(), dtype=dtype).reshape(shape)
    actual = pp.euler2SO3(angles)
    assert actual.shape == angles.shape[:-1] + (4,)
    torch.testing.assert_close(actual.matrix(), rotation_matrix(angles))


@pytest.mark.parametrize('factory', (list, tuple, np.asarray))
def test_array_like_angles_keep_existing_conversion(factory):
    angles = factory([[0.1, -0.2, 0.3], [0.4, 0.5, -0.6]])
    actual = pp.euler2SO3(angles)
    expected = rotation_matrix(torch.tensor(angles))
    torch.testing.assert_close(actual.matrix(), expected)


def test_strided_conversion_has_correct_first_and_second_derivatives():
    angles = torch.randn(2, 2, 3, dtype=torch.float64, requires_grad=True)
    function = lambda values: pp.euler2SO3(values.transpose(0, 1)).tensor()
    assert torch.autograd.gradcheck(function, (angles,))
    assert torch.autograd.gradgradcheck(function, (angles,))


def test_expanded_angles_remain_usable_for_point_rotation():
    angles = torch.tensor([[[0.0, 0.0, torch.pi / 2], [torch.pi / 2, 0.0, 0.0]]], dtype=torch.float64)
    angles = angles.expand(3, 2, 3)
    points = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    actual = pp.euler2SO3(angles).Act(points)
    expected = (rotation_matrix(angles) @ points.unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(actual, expected)
