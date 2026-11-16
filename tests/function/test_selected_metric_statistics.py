import math
import warnings

import pytest
import torch
import pypose as pp
from pypose.metric.ape_rpe import StampedSE3, compute_error


STATISTICS = ('Max', 'Min', 'Mean', 'Median', 'RMSE', 'SSE', 'STD')


def reduce_error(error, statistic):
    values = {
        'Max': lambda: error.max(),
        'Min': lambda: error.min(),
        'Mean': lambda: error.mean(),
        'Median': lambda: error.median(),
        'RMSE': lambda: error.square().mean().sqrt(),
        'SSE': lambda: error.square().sum(),
        'STD': lambda: error.std(),
    }
    return values[statistic]()


def trajectory(position):
    angle = 0.1 * position.square()
    zeros = torch.zeros_like(position)
    rotation = pp.so3(torch.stack((angle, zeros, zeros), dim=-1)).Exp()
    translation = torch.stack((position, zeros, zeros), dim=-1)
    return pp.SE3(torch.cat((translation, rotation.tensor()), dim=-1))


@pytest.mark.parametrize('metric', ('ape', 'rpe'))
@pytest.mark.parametrize('statistic', STATISTICS[:-1])
def test_single_error_does_not_compute_unrequested_standard_deviation(metric, statistic):
    count = 1 if metric == 'ape' else 2
    reference = pp.identity_SE3(count, dtype=torch.float64)
    estimate = reference.clone()
    estimate[-1, 0] = 0.25
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        result = getattr(pp.metric, metric)(None, reference, None, estimate,
                                            otype=statistic)
    expected = 0.25 ** 2 if statistic == 'SSE' else 0.25
    torch.testing.assert_close(result, result.new_tensor(expected))


@pytest.mark.parametrize('metric', ('ape', 'rpe'))
@pytest.mark.parametrize('etype', ('translation', 'rotation', 'pose', 'radian', 'degree'))
@pytest.mark.parametrize('statistic', STATISTICS + ('All',))
def test_requested_statistics_match_independent_trajectory_formulas(metric, etype, statistic):
    position = torch.tensor([0.1, 0.3, 0.7, 1.2, 1.8], dtype=torch.float64)
    angle = 0.1 * position.square()
    reference = pp.identity_SE3(len(position), dtype=position.dtype)
    estimate = trajectory(position)
    if metric == 'rpe':
        position, angle = position.diff(), angle.diff()
    if etype == 'translation':
        error = position
    elif etype == 'rotation':
        error = 2 * math.sqrt(2) * torch.sin(angle / 2)
    elif etype == 'pose':
        error = torch.sqrt(position.square() + 8 * torch.sin(angle / 2).square())
    elif etype == 'radian':
        error = angle
    else:
        error = angle * (180 / math.pi)
    result = getattr(pp.metric, metric)(None, reference, None, estimate,
                                        etype=etype, otype=statistic)
    if statistic == 'All':
        assert tuple(result) == STATISTICS
        for name in STATISTICS:
            torch.testing.assert_close(result[name], reduce_error(error, name))
    else:
        assert isinstance(result, torch.Tensor)
        torch.testing.assert_close(result, reduce_error(error, statistic))


@pytest.mark.parametrize('metric', ('ape', 'rpe'))
@pytest.mark.parametrize('statistic', STATISTICS)
@pytest.mark.parametrize('dtype', (torch.float32, torch.float64))
def test_selected_statistic_retains_position_gradient(metric, statistic, dtype):
    position = torch.tensor([0.1, 0.3, 0.7, 1.2, 1.8], dtype=dtype,
                            requires_grad=True)
    error = position.double()
    if metric == 'rpe':
        error = error.diff()
    reference = StampedSE3(None, pp.identity_SE3(len(error), dtype=dtype))
    estimate = StampedSE3(None, trajectory(error))
    result = compute_error(reference, estimate, mtype=metric, otype=statistic)
    expected = reduce_error(error, statistic)
    actual_gradient, = torch.autograd.grad(result, position, retain_graph=True)
    expected_gradient, = torch.autograd.grad(expected, position)
    torch.testing.assert_close(result, expected)
    torch.testing.assert_close(actual_gradient, expected_gradient)


@pytest.mark.parametrize('metric', ('ape', 'rpe'))
@pytest.mark.parametrize('statistic', ('STD', 'All'))
def test_requested_single_sample_std_keeps_existing_nan_and_warning(metric, statistic):
    count = 1 if metric == 'ape' else 2
    reference = pp.identity_SE3(count, dtype=torch.float64)
    estimate = reference.clone()
    estimate[-1, 0] = 0.25
    with pytest.warns(UserWarning, match='degrees of freedom'):
        result = getattr(pp.metric, metric)(None, reference, None, estimate,
                                            otype=statistic)
    assert torch.isnan(result if statistic == 'STD' else result['STD'])


@pytest.mark.parametrize('metric', ('ape', 'rpe'))
def test_unknown_statistic_is_rejected(metric):
    poses = pp.identity_SE3(3, dtype=torch.float64)
    with pytest.raises(ValueError, match='Unknown output metric type'):
        getattr(pp.metric, metric)(None, poses, None, poses, otype='Variance')
