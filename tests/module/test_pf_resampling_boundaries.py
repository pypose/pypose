from unittest.mock import patch

import pytest
import torch
import pypose as pp


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("batch", [(), (2,), (2, 1)])
def test_half_open_weight_intervals(dtype, batch):
    # The two positive intervals are [0, .25) and [.25, 1).
    weights = torch.tensor([0., .25, 0., .75, 0.], dtype=dtype)
    weights = weights.reshape(5, *(1 for _ in batch)).expand(5, *batch)
    values = torch.arange(5, dtype=dtype).reshape(5, *(1 for _ in batch), 1)
    values = values.expand(5, *batch, 2)
    draws = torch.tensor([0., .125, .25, .5, .875], dtype=dtype)
    draws = draws.reshape(5, *(1 for _ in batch)).expand(5, *batch)
    expected_indices = torch.tensor([1, 1, 3, 3, 3])
    expected = expected_indices.to(dtype).reshape(5, *(1 for _ in batch), 1)
    expected = expected.expand_as(values)
    pf = pp.module.PF(pp.module.NLS(), particles=5)
    with patch("torch.rand", return_value=draws):
        actual = pf.resample_particles(weights, values)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_different_distributions_in_one_batch(dtype):
    weights = torch.tensor([[0., .5], [.5, 0.], [0., .5], [.5, 0.]], dtype=dtype)
    values = torch.tensor([0., 1., 2., 3.], dtype=dtype).reshape(4, 1, 1)
    values = values.expand(4, 2, 1).clone()
    values[:, 1] += 10
    draws = torch.tensor([[0., 0.], [.5, .5], [.25, .25], [.75, .75]], dtype=dtype)
    expected = torch.tensor([[1., 10.], [3., 12.], [1., 10.], [3., 12.]],
                            dtype=dtype).unsqueeze(-1)
    pf = pp.module.PF(pp.module.NLS(), particles=4)
    with patch("torch.rand", return_value=draws):
        actual = pf.resample_particles(weights, values)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_native_draws_never_select_zero_weight(dtype, seed):
    weights = torch.zeros(4096, dtype=dtype)
    weights[-1] = 1
    values = torch.zeros(4096, 1, dtype=dtype)
    values[-1] = 1
    pf = pp.module.PF(pp.module.NLS(), particles=4096)
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        actual = pf.resample_particles(weights, values)
    assert torch.equal(actual, torch.ones_like(actual))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_interior_draws_keep_existing_assignments(dtype):
    weights = torch.tensor([.25, .25, .5], dtype=dtype)
    values = torch.tensor([[2., 3.], [5., 7.], [11., 13.]], dtype=dtype)
    draws = torch.tensor([.125, .375, .875], dtype=dtype)
    pf = pp.module.PF(pp.module.NLS(), particles=3)
    with patch("torch.rand", return_value=draws):
        actual = pf.resample_particles(weights, values)
    torch.testing.assert_close(actual, values)
