import math
import pytest
import torch
import pypose as pp


GROUPS = (pp.SO3, pp.SE3, pp.RxSO3, pp.Sim3)


def make_group(constructor, quaternion):
    if constructor in (pp.SE3, pp.Sim3):
        translation = torch.tensor([0.25, -0.5, 0.75], dtype=quaternion.dtype).expand(quaternion.shape[:-1] + (3,))
        values = torch.cat((translation, quaternion), -1)
    else:
        values = quaternion
    if constructor in (pp.RxSO3, pp.Sim3):
        values = torch.cat((values, torch.full_like(values[..., :1], 2.)), -1)
    return constructor(values)


def quaternion_slice(constructor):
    return slice(3, 7) if constructor in (pp.SE3, pp.Sim3) else slice(0, 4)


def reference_normalization(quaternion, eps):
    rows = quaternion.double().reshape(-1, 4).tolist()
    normalized = [[value / max(math.hypot(*row), eps) for value in row] for row in rows]
    return torch.tensor(normalized, dtype=quaternion.dtype).reshape(quaternion.shape)


@pytest.mark.parametrize('constructor', GROUPS)
@pytest.mark.parametrize('dtype,magnitude', (
    (torch.float16, 40000.), (torch.bfloat16, 1e30),
    (torch.float32, 1e20), (torch.float32, 1e30),
    (torch.float64, 1e200), (torch.float64, 1e300),
))
def test_large_finite_quaternion_directions_are_preserved(constructor, dtype, magnitude):
    quaternion = torch.tensor([[1., -1., 1., 1.], [-0.5, 1., -0.25, 0.75]], dtype=dtype) * magnitude
    group = make_group(constructor, quaternion)
    original = group.tensor().clone()
    actual = pp.quat2unit(group)
    expected = original.clone()
    expected[..., quaternion_slice(constructor)] = reference_normalization(quaternion, 1e-12)
    assert actual.ltype is group.ltype
    assert actual.shape == group.shape and actual.dtype == group.dtype
    assert torch.isfinite(actual.tensor()).all()
    torch.testing.assert_close(actual.tensor(), expected)
    torch.testing.assert_close(group.tensor(), original, atol=0, rtol=0)
    torch.testing.assert_close(actual.rotation().tensor().float().norm(dim=-1),
                               torch.ones(2), atol=4e-3 if dtype in (torch.float16, torch.bfloat16) else 2e-7, rtol=0)


@pytest.mark.parametrize('constructor', GROUPS)
@pytest.mark.parametrize('magnitude,eps', ((1., 1e-12), (2., 1e-12), (1e-8, 1e-6), (0.02, 0.2)))
def test_ordinary_values_and_epsilon_floor_keep_documented_formula(constructor, magnitude, eps):
    quaternion = torch.tensor([1., -2., 3., 4.], dtype=torch.float64) * magnitude
    group = make_group(constructor, quaternion)
    expected = group.tensor().clone()
    expected[..., quaternion_slice(constructor)] = reference_normalization(quaternion, eps)
    actual = pp.quat2unit(group, eps=eps)
    torch.testing.assert_close(actual.tensor(), expected, atol=2e-16, rtol=2e-15)


@pytest.mark.parametrize('constructor', GROUPS)
@pytest.mark.parametrize('dtype,magnitude', ((torch.float32, 1e30), (torch.float64, 1e300)))
def test_large_input_gradients_match_analytic_normalization(constructor, dtype, magnitude):
    values = torch.tensor([1., -2., 3., 4.], dtype=dtype) * magnitude
    values.requires_grad_()
    weights = torch.tensor([0.3, -0.7, 0.2, 0.4], dtype=dtype)
    output = pp.quat2unit(make_group(constructor, values)).tensor()[quaternion_slice(constructor)]
    actual = torch.autograd.grad((output * weights).sum(), values)[0]
    row = values.detach().double().tolist()
    norm = math.hypot(*row)
    direction = torch.tensor([x / norm for x in row], dtype=torch.float64)
    weights64 = weights.double()
    expected = ((weights64 - (weights64 * direction).sum() * direction) / norm).to(dtype)
    assert torch.isfinite(actual).all() and actual.abs().max() > 0
    torch.testing.assert_close(actual, expected, atol=0, rtol=2e-5 if dtype == torch.float32 else 1e-13)


@pytest.mark.parametrize('constructor', GROUPS)
@pytest.mark.parametrize('dtype', (torch.float32, torch.float64))
def test_zero_quaternions_keep_existing_error(constructor, dtype):
    with pytest.raises(ValueError, match='zero quaternions'):
        pp.quat2unit(make_group(constructor, torch.zeros(4, dtype=dtype)))


@pytest.mark.parametrize('constructor', GROUPS)
def test_empty_batch_keeps_lie_type_and_shape(constructor):
    group = make_group(constructor, torch.empty(2, 0, 4, dtype=torch.float64))
    actual = pp.quat2unit(group)
    assert actual.ltype is group.ltype and actual.shape == group.shape


@pytest.mark.parametrize('values', ([1., 1., 1., 1.], [2., -2., 1., 0.5]))
def test_scale_ties_preserve_first_and_second_derivatives(values):
    quaternion = torch.tensor(values, dtype=torch.float64, requires_grad=True)
    function = lambda data: pp.quat2unit(pp.SO3(data)).tensor()
    assert torch.autograd.gradcheck(function, (quaternion,))
    assert torch.autograd.gradgradcheck(function, (quaternion,))
