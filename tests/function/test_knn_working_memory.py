import pytest
import torch
import pypose as pp
import pypose.function.geometry as geometry


@pytest.fixture(autouse=True)
def small_working_set(monkeypatch):
    monkeypatch.setattr(geometry, '_KNN_WORKING_BYTES', 512, raising=False)


def dense_reference(reference, neighbors, k=1, ord=2, dim=-1, largest=False, sorted=True):
    distances = torch.linalg.norm(reference.unsqueeze(-2) - neighbors.unsqueeze(-3),
                                   ord=ord, dim=dim)
    return distances.topk(k, dim=dim, largest=largest, sorted=sorted)


def assert_same(actual, expected):
    assert type(actual) is type(expected)
    torch.testing.assert_close(actual.values, expected.values, atol=0, rtol=0, equal_nan=True)
    torch.testing.assert_close(actual.indices, expected.indices)


@pytest.mark.parametrize('order', (0, 1, 2, float('inf'), -float('inf')))
@pytest.mark.parametrize('k', (0, 1, 4))
@pytest.mark.parametrize('largest', (False, True))
@pytest.mark.parametrize('sorted', (False, True))
def test_broadcast_search_matches_dense_norm_and_topk(order, k, largest, sorted):
    generator = torch.Generator().manual_seed(502)
    reference = torch.randn(2, 1, 17, 3, generator=generator, dtype=torch.float32)
    neighbors = torch.randn(1, 3, 19, 3, generator=generator, dtype=torch.float64)
    original_reference, original_neighbors = reference.clone(), neighbors.clone()
    options = dict(k=k, ord=order, largest=largest, sorted=sorted)
    assert_same(pp.knn(reference, neighbors, **options), dense_reference(reference, neighbors, **options))
    torch.testing.assert_close(reference, original_reference)
    torch.testing.assert_close(neighbors, original_neighbors)


@pytest.mark.parametrize('dtype', (torch.float16, torch.bfloat16, torch.float32,
                                  torch.float64, torch.complex64, torch.complex128))
def test_strided_coordinate_views_keep_results(dtype):
    generator = torch.Generator().manual_seed(503)
    reference = torch.randn(34, 10, generator=generator, dtype=dtype)[::2, ::2]
    neighbors = torch.randn(38, 10, generator=generator, dtype=dtype)[::2, ::2]
    assert_same(pp.knn(reference, neighbors, k=3), dense_reference(reference, neighbors, k=3))


@pytest.mark.parametrize('largest', (False, True))
@pytest.mark.parametrize('sorted', (False, True))
def test_complete_neighbor_rows_preserve_equal_distance_indices(largest, sorted):
    reference = torch.zeros(17, 2, dtype=torch.float64)
    neighbors = torch.tensor([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=torch.float64).repeat(5, 1)
    options = dict(k=7, largest=largest, sorted=sorted)
    assert_same(pp.knn(reference, neighbors, **options), dense_reference(reference, neighbors, **options))


@pytest.mark.parametrize('reference_shape,neighbor_shape', (
    ((0, 3), (19, 3)), ((0, 17, 3), (1, 19, 3)), ((17, 1), (19, 3)),
    ((1, 3), (19, 3)), ((17, 0), (19, 0)),
))
def test_empty_and_singleton_axes_keep_original_behavior(reference_shape, neighbor_shape):
    reference, neighbors = torch.zeros(reference_shape), torch.ones(neighbor_shape)
    assert_same(pp.knn(reference, neighbors), dense_reference(reference, neighbors))


def test_mixed_half_precision_uses_promoted_difference_dtype():
    reference = torch.arange(51, dtype=torch.float16).reshape(17, 3)
    neighbors = torch.arange(57, dtype=torch.bfloat16).reshape(19, 3)
    assert_same(pp.knn(reference, neighbors, k=2), dense_reference(reference, neighbors, k=2))


@pytest.mark.parametrize('width', (1, 3))
def test_requesting_all_neighbors_preserves_dense_result(width):
    reference, neighbors = torch.randn(17, width), torch.randn(19, width)
    assert_same(pp.knn(reference, neighbors, k=19), dense_reference(reference, neighbors, k=19))


def test_nonfinite_rows_keep_topk_behavior():
    reference = torch.zeros(17, 3, dtype=torch.float64)
    neighbors = torch.arange(57, dtype=torch.float64).reshape(19, 3)
    reference[2, 0] = float('nan')
    reference[8, 1] = float('inf')
    neighbors[4, 2] = float('inf')
    assert_same(pp.knn(reference, neighbors, k=3), dense_reference(reference, neighbors, k=3))


@pytest.mark.parametrize('order', (1, 2))
def test_broadcast_input_gradients_match_dense_reference(order):
    generator = torch.Generator().manual_seed(504)
    reference = torch.randn(2, 1, 17, 3, generator=generator, dtype=torch.float64, requires_grad=True)
    neighbors = torch.randn(1, 3, 19, 3, generator=generator, dtype=torch.float64, requires_grad=True)
    actual = pp.knn(reference, neighbors, k=3, ord=order)
    expected = dense_reference(reference, neighbors, k=3, ord=order)
    assert_same(actual, expected)
    actual_gradients = torch.autograd.grad(actual.values.square().sum(), (reference, neighbors))
    expected_gradients = torch.autograd.grad(expected.values.square().sum(), (reference, neighbors))
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual_gradient, expected_gradient, atol=1e-11, rtol=1e-11)


def test_nondefault_dimension_retains_existing_behavior():
    reference, neighbors = torch.randn(2, 17, 3), torch.randn(2, 19, 3)
    assert_same(pp.knn(reference, neighbors, dim=-2), dense_reference(reference, neighbors, dim=-2))


def test_public_icp_with_small_working_set_recovers_known_pose():
    generator = torch.Generator().manual_seed(505)
    source = torch.randn(80, 3, generator=generator, dtype=torch.float64) * 4
    transform = pp.se3(torch.tensor([0.002, -0.003, 0.001, 0, 0, 0], dtype=torch.float64)).Exp()
    target = transform.Act(source)
    icp = pp.module.ICP(stepper=pp.utils.ReduceToBason(steps=3))
    actual = icp(source, target)
    torch.testing.assert_close(actual.matrix(), transform.matrix(), atol=1e-10, rtol=1e-10)
