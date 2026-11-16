import pytest
import torch
import pypose as pp

from pypose.metric.ape_rpe import matching_time_indices


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('offset', [-0.5, 0.0, 0.5])
def test_matching_keeps_inputs_and_repeated_results(dtype, offset):
    reference = torch.tensor([0.0, 1.0, 2.0, 4.0], dtype=dtype)
    estimate = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=dtype) - offset
    reference_before, estimate_before = reference.clone(), estimate.clone()
    expected = ([0, 1, 2], [0, 1, 2])
    for _ in range(2):
        assert matching_time_indices(reference, estimate, offset_2=offset) == expected
        torch.testing.assert_close(reference, reference_before, rtol=0, atol=0)
        torch.testing.assert_close(estimate, estimate_before, rtol=0, atol=0)


@pytest.mark.parametrize('offset', [-0.5, 0.0, 0.5])
def test_matching_accepts_leaf_timestamp_tensors(offset):
    reference = torch.arange(4, dtype=torch.float64, requires_grad=True)
    estimate = (reference.detach() - offset).requires_grad_()
    before = estimate.detach().clone()
    assert matching_time_indices(reference, estimate, offset_2=offset) == (
        [0, 1, 2, 3], [0, 1, 2, 3]
    )
    torch.testing.assert_close(estimate, before, rtol=0, atol=0)
    assert estimate.grad is None and reference.grad is None


def test_matching_does_not_modify_noncontiguous_aliases():
    storage = torch.tensor([0.0, 91.0, 1.0, 92.0, 2.0, 93.0], dtype=torch.float64)
    before = storage.clone()
    estimate = storage[::2]
    reference = torch.tensor([0.25, 1.25, 2.25], dtype=torch.float64)
    assert matching_time_indices(reference, estimate, offset_2=0.25) == (
        [0, 1, 2], [0, 1, 2]
    )
    torch.testing.assert_close(storage, before, rtol=0, atol=0)


@pytest.mark.parametrize('metric', [pp.metric.ape, pp.metric.rpe])
@pytest.mark.parametrize('reference_count,estimate_count', [(6, 6), (6, 8), (8, 6)])
@pytest.mark.parametrize('offset', [-0.25, 0.25])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_public_metrics_preserve_timestamps_and_repeat(metric, reference_count, estimate_count,
                                                       offset, dtype):
    reference_stamps = torch.arange(reference_count, dtype=dtype)
    estimate_stamps = torch.arange(estimate_count, dtype=dtype) - offset
    reference_before, estimate_before = reference_stamps.clone(), estimate_stamps.clone()
    reference_poses = pp.identity_SE3(reference_count, dtype=torch.float64)
    estimate_poses = pp.identity_SE3(estimate_count, dtype=torch.float64)
    reference_poses.tensor()[:, 0] = torch.arange(reference_count, dtype=torch.float64)
    positions = torch.arange(estimate_count, dtype=torch.float64)
    estimate_poses.tensor()[:, 0] = positions + 0.1 * positions.square()
    matched = min(reference_count, estimate_count)
    if metric is pp.metric.ape:
        expected = 0.1 * sum(i * i for i in range(matched)) / matched
    else:
        expected = 0.1 * (matched - 1)
    for _ in range(2):
        result = metric(reference_stamps, reference_poses, estimate_stamps,
                        estimate_poses, offset=offset, otype='Mean')
        torch.testing.assert_close(result, torch.tensor(expected, dtype=torch.float64))
        torch.testing.assert_close(reference_stamps, reference_before, rtol=0, atol=0)
        torch.testing.assert_close(estimate_stamps, estimate_before, rtol=0, atol=0)


def test_matching_keeps_nearest_ties_and_strict_threshold():
    reference = torch.tensor([1.0, 3.0, 4.0], dtype=torch.float64)
    estimate = torch.tensor([2.0, 0.0, 3.0], dtype=torch.float64)
    assert matching_time_indices(reference, estimate, max_diff=1.1) == ([0, 1, 2], [0, 2, 2])
    assert matching_time_indices(reference, estimate, max_diff=1.0) == ([1], [2])
