import pytest
import torch
import pypose as pp


def point_groups(counts, dimensions, dtype):
    groups = []
    for index, count in enumerate(counts):
        offset = torch.linspace(0., 0.25, count, dtype=torch.float64)
        coordinates = [2 * index + 0.125 + offset]
        coordinates += [0.25 + 0.1 * offset for _ in range(dimensions - 1)]
        attributes = [torch.linspace(900., 1100., count, dtype=torch.float64),
                      100 + 50 * index + torch.sin(torch.arange(count, dtype=torch.float64))]
        groups.append(torch.stack(coordinates + attributes, dim=-1).to(dtype))
    return groups


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("counts", [(256,), (4096,), (3000, 2000, 12)])
@pytest.mark.parametrize("dimensions", [1, 3])
def test_voxel_centroids_match_independent_high_precision_group_means(dtype, counts, dimensions):
    groups = point_groups(counts, dimensions, dtype)
    points = torch.cat(groups)
    original = points.clone()

    actual = pp.voxel_filter(points, [1.] * dimensions)

    expected = torch.stack([group.double().mean(0).to(dtype) for group in groups])
    assert actual.dtype == dtype
    assert actual.shape == (len(counts), dimensions + 2)
    assert actual.isfinite().all()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(points, original, rtol=0, atol=0)
    for result, group in zip(actual, groups):
        assert torch.all(result >= group.amin(0))
        assert torch.all(result <= group.amax(0))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("strided", [False, True])
def test_centroid_gradients_use_actual_voxel_counts(dtype, strided):
    counts = (3000, 800)
    points = torch.cat(point_groups(counts, 3, dtype))
    if strided:
        backing = torch.empty(points.shape[0], points.shape[1] * 2, dtype=dtype)
        backing[:, ::2] = points
        points = backing[:, ::2]
        assert not points.is_contiguous()
    points.requires_grad_()

    actual = pp.voxel_filter(points, [1., 1., 1.])
    actual.sum().backward()

    expected = torch.cat([torch.full((count, points.shape[1]), 1 / count, dtype=dtype)
                          for count in counts])
    torch.testing.assert_close(points.grad, expected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_large_single_voxel_feature_average_stays_inside_input_range(dtype):
    points = torch.zeros(4096, 4, dtype=dtype)
    points[:, 3] = torch.linspace(0., 100., 4096, dtype=torch.float64).to(dtype)

    actual = pp.voxel_filter(points, [1., 1., 1.])

    torch.testing.assert_close(actual[0], points.double().mean(0).to(dtype))
    assert 0 <= actual[0, 3] <= 100


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_random_representatives_preserve_seeded_selection(dtype):
    groups = point_groups((4, 7, 5), 3, dtype)
    points = torch.cat(groups)
    # All grouping indices are already in sorted voxel order.
    with torch.random.fork_rng():
        torch.manual_seed(13)
        expected = torch.stack([group[torch.randint(len(group), (1,)).item()] for group in groups])
        torch.manual_seed(13)
        actual = pp.voxel_filter(points, [1., 1., 1.], random=True)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
