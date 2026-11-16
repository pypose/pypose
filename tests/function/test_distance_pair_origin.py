import math

import pytest
import torch
import pypose as pp
from pypose.metric.ape_rpe import StampedSE3, pairs_by_dist


def poses_at(positions, dtype):
    coordinates = torch.tensor(positions, dtype=dtype)
    result = pp.identity_SE3(len(positions), dtype=dtype)
    if coordinates.ndim == 1:
        result[:, 0] = coordinates
    else:
        result[:, :3] = coordinates
    return result


@pytest.mark.parametrize('dtype', (torch.float32, torch.float64))
@pytest.mark.parametrize('positions,delta,expected', (
    ([0, 1], 1, ([0], [1])),
    ([0, 1, 2], 1, ([0, 1], [1, 2])),
    ([0, 0, 1, 1, 2], 1, ([0, 2], [2, 4])),
    ([(0, 0, 0), (0.6, 0, 0), (0.6, 0.8, 0), (1.2, 0.8, 0)],
     1, ([0], [2])),
    ([0, 1.7, 2.2, 2.8, 3], 1, ([0, 1], [1, 3])),
    ([0, 0.25, 0.5, 0.75, 1], 0.5, ([0, 2], [2, 4])),
    ([0, 0.1], 1, ([], [])),
    ([0, 0], 1, ([], [])),
    ([0], 1, ([], [])),
    ([0, 1, 0, 1], 1, ([0, 1, 2], [1, 2, 3])),
))
def test_consecutive_distance_pairs_include_initial_pose(positions, delta, expected, dtype):
    poses = poses_at(positions, dtype)
    original = poses.clone()
    trajectory = StampedSE3(None, poses)
    assert pairs_by_dist(trajectory, delta) == expected
    torch.testing.assert_close(poses, original)


@pytest.mark.parametrize('dtype', (torch.float32, torch.float64))
@pytest.mark.parametrize('rpair', (False, True))
@pytest.mark.parametrize('statistic,expected', (
    ('Mean', 0.5), ('SSE', 1.0), ('RMSE', math.sqrt(0.5)), ('Max', 1.0),
))
def test_public_rpe_includes_first_segment_error(dtype, rpair, statistic, expected):
    reference = poses_at([0, 1, 2], dtype)
    estimate = poses_at([0, 2, 3], dtype)
    result = pp.metric.rpe(None, reference, None, estimate, associate='distance',
                           delta=1.0, rpair=rpair, otype=statistic)
    torch.testing.assert_close(result, torch.tensor(expected, dtype=torch.float64))


@pytest.mark.parametrize('dtype', (torch.float32, torch.float64))
@pytest.mark.parametrize('rpair', (False, True))
def test_public_rpe_accepts_one_valid_distance_pair(dtype, rpair):
    reference = poses_at([0, 1], dtype)
    estimate = poses_at([0, 2], dtype)
    result = pp.metric.rpe(None, reference, None, estimate, associate='distance',
                           delta=1.0, rpair=rpair, otype='SSE')
    torch.testing.assert_close(result, result.new_tensor(1.0))


@pytest.mark.parametrize('dtype', (torch.float32, torch.float64))
@pytest.mark.parametrize('associate,all_pairs', (('frame', False), ('distance', True)))
def test_other_pairing_modes_keep_existing_result(dtype, associate, all_pairs):
    reference = poses_at([0, 1, 2], dtype)
    estimate = poses_at([0, 2, 3], dtype)
    result = pp.metric.rpe(None, reference, None, estimate, associate=associate,
                           all=all_pairs, delta=1.0, rpair=True, otype='Mean')
    torch.testing.assert_close(result, result.new_tensor(0.5))
