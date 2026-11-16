import pytest
import torch
import pypose as pp


def cubic_weights(t):
    return torch.stack(((1-t)**3, 3*t**3-6*t**2+4,
                        -3*t**3+3*t**2+3*t+1, t**3), dim=-1) / 6


@pytest.mark.parametrize('batch', ((), (2,), (2, 1)))
@pytest.mark.parametrize('count,extrapolate', ((4, False), (8, False), (1, True),
                                             (2, True), (8, True)))
@pytest.mark.parametrize('interval', (0.5, 0.3))
@pytest.mark.parametrize('dtype', (torch.float32, torch.float64))
def test_commuting_pose_spline_matches_cubic_basis(batch, count, extrapolate, interval, dtype):
    position = torch.arange(count, dtype=dtype).square() * 0.2
    angle = torch.arange(count, dtype=dtype).square() * 0.02
    coordinates = torch.zeros(batch + (count, 6), dtype=dtype)
    coordinates[..., 2] = position
    coordinates[..., 5] = angle
    controls = pp.se3(coordinates).Exp()
    original = controls.clone()
    result = pp.bspline(controls, interval, extrapolate)
    if extrapolate:
        position = torch.cat((position[:1].repeat(2), position, position[-1:].repeat(2)))
        angle = torch.cat((angle[:1].repeat(2), angle, angle[-1:].repeat(2)))
    samples = torch.arange(0, 1, interval, dtype=dtype)
    weights = cubic_weights(samples)
    z, theta = [], []
    for i in range(len(position)-3):
        z.append(weights @ position[i:i+4])
        theta.append(weights @ angle[i:i+4])
    end_weights = cubic_weights(torch.ones(1, dtype=dtype))
    z.append(end_weights @ position[-4:])
    theta.append(end_weights @ angle[-4:])
    z, theta = torch.cat(z), torch.cat(theta)
    expected = torch.eye(4, dtype=dtype).expand(len(z), 4, 4).clone()
    expected[:, 0, 0] = expected[:, 1, 1] = theta.cos()
    expected[:, 0, 1], expected[:, 1, 0] = -theta.sin(), theta.sin()
    expected[:, 2, 3] = z
    expected = expected.expand(batch + expected.shape)
    torch.testing.assert_close(result.matrix(), expected, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(controls, original)


@pytest.mark.parametrize('extrapolate', (False, True))
def test_noncontiguous_pose_controls_match_contiguous_inputs(extrapolate):
    generator = torch.Generator().manual_seed(187)
    controls = pp.se3(torch.randn(2, 16, 6, generator=generator, dtype=torch.float64) * 0.2).Exp()
    view = controls[:, ::2]
    assert not view.is_contiguous()
    actual = pp.bspline(view, 0.3, extrapolate)
    expected = pp.bspline(pp.SE3(view.tensor().contiguous()), 0.3, extrapolate)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize('count,interval,extrapolate', ((4, 0.5, False),
                                                      (6, 0.3, False),
                                                      (2, 0.5, True)))
def test_spline_input_gradient_matches_finite_difference(count, interval, extrapolate):
    generator = torch.Generator().manual_seed(18)
    coordinates = (torch.randn(count, 6, generator=generator, dtype=torch.float64) * 0.2).requires_grad_()
    probe = torch.tensor([0.2, -0.4, 0.7], dtype=torch.float64)

    def objective(values):
        interpolated = pp.bspline(pp.se3(values).Exp(), interval, extrapolate)
        return interpolated.Act(probe).square().sum()

    gradient, = torch.autograd.grad(objective(coordinates), coordinates)
    numerical = torch.empty_like(coordinates)
    step = 1e-6
    for i in range(coordinates.numel()):
        positive, negative = coordinates.detach().clone(), coordinates.detach().clone()
        positive.view(-1)[i] += step
        negative.view(-1)[i] -= step
        numerical.view(-1)[i] = (objective(positive) - objective(negative)) / (2 * step)
    torch.testing.assert_close(gradient, numerical, atol=2e-7, rtol=2e-6)
