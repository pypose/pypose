import pytest
import torch
import pypose as pp

from pypose.metric.ape_rpe import matching_time_indices, _nearest_sorted_timestamps


def dense_reference(query, reference, threshold):
    distances = (query[:, None] - reference[None]).abs()
    values, indices = distances.min(dim=-1)
    accepted = values < threshold
    return torch.arange(len(query))[accepted].tolist(), indices[accepted].tolist()


def ordered_reference_check(query, reference):
    expected_values, expected_indices = (query[:, None] - reference[None]).abs().min(-1)
    values, indices = _nearest_sorted_timestamps(query, reference)
    torch.testing.assert_close(values, expected_values, rtol=0, atol=0)
    torch.testing.assert_close(indices, expected_indices)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize('count', [1, 2, 7, 32])
def test_sorted_search_matches_dense_nearest_indices(dtype, count):
    for seed in range(8):
        generator = torch.Generator().manual_seed(seed)
        query = (torch.randn(31, generator=generator) * 5).to(dtype)
        reference = torch.randint(-5, 6, (count,), generator=generator).to(dtype).sort().values
        ordered_reference_check(query, reference)
        for threshold in [0.0, 0.25, 1.0, float('inf')]:
            actual = matching_time_indices(query, reference.clone(), max_diff=threshold)
            assert actual == dense_reference(query, reference, threshold)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_rounding_plateaus_keep_the_first_index(dtype):
    query = torch.tensor([1000.0, -1000.0, 10000.0], dtype=dtype)
    reference = torch.tensor([-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3], dtype=dtype)
    ordered_reference_check(query, reference)
    assert matching_time_indices(query, reference.clone(), max_diff=20000) == (
        dense_reference(query, reference, 20000)
    )


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_duplicates_noncontiguous_inputs_and_threshold_ties(dtype):
    reference = torch.tensor([-2, 99, -2, 99, 0, 99, 2, 99, 2, 99, 4, 99], dtype=dtype)[::2]
    query = torch.tensor([-1, 99, 1, 99, 3, 99, 4, 99], dtype=dtype)[::2]
    ordered_reference_check(query, reference)
    for threshold in [1.0, 1.01]:
        assert matching_time_indices(query, reference.clone(), max_diff=threshold) == (
            dense_reference(query, reference, threshold)
        )


@pytest.mark.parametrize('query_dtype,reference_dtype', [
    (torch.float32, torch.float64), (torch.float64, torch.float32)
])
def test_mixed_timestamp_dtypes(query_dtype, reference_dtype):
    query = torch.tensor([0.100000002, 0.200000005, 0.299999999], dtype=query_dtype)
    reference = torch.tensor([0.1, 0.2, 0.3], dtype=reference_dtype)
    ordered_reference_check(query, reference)
    assert matching_time_indices(query, reference.clone()) == dense_reference(query, reference, 0.01)


@pytest.mark.parametrize('query,reference', [
    ([1.0, 3.0], [2.0, 0.0, 3.0]),
    ([1.0, 2.0], [float('nan'), 1.0, 2.0]),
    ([float('nan'), 1.0, float('inf')], [0.0, 1.0, 2.0]),
    ([float('-inf'), 1.0, float('inf')], [float('-inf'), 1.0, float('inf')]),
])
def test_unsorted_and_nonfinite_inputs_keep_general_results(query, reference):
    query, reference = torch.tensor(query), torch.tensor(reference)
    assert matching_time_indices(query, reference.clone(), max_diff=2.0) == (
        dense_reference(query, reference, 2.0)
    )


def test_empty_query_keeps_empty_matches():
    assert matching_time_indices(torch.empty(0), torch.arange(4, dtype=torch.float32)) == ([], [])


@pytest.mark.parametrize('metric', [pp.metric.ape, pp.metric.rpe])
def test_public_trajectory_metrics_keep_their_values(metric):
    count = 600
    stamps = torch.arange(count, dtype=torch.float64) * 0.01
    reference = pp.identity_SE3(count, dtype=torch.float64)
    estimate = pp.identity_SE3(count, dtype=torch.float64)
    reference.tensor()[:, 0] = torch.arange(count, dtype=torch.float64)
    estimate.tensor()[:, 0] = 1.1 * torch.arange(count, dtype=torch.float64)
    result = metric(stamps.clone(), reference, stamps + 0.001, estimate,
                    diff=0.002, otype='Mean')
    expected = 0.1 * (count - 1) / 2 if metric is pp.metric.ape else 0.1
    torch.testing.assert_close(result, torch.tensor(expected, dtype=torch.float64))
