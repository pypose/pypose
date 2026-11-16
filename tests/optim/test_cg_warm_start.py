import pytest
import torch
from pypose.optim.solver import CG


def system(dtype):
    A = torch.tensor([[5., 1., 0., 0.], [1., 4., 1., 0.],
                      [0., 1., 3., 1.], [0., 0., 1., 2.]], dtype=dtype)
    b = torch.tensor([1., -2., 3., .5], dtype=dtype)
    return A, b


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("layout", ["dense", "coo", "csr", "bsr"])
@pytest.mark.parametrize("guess", ["none", "zero", "nonzero", "exact", "column"])
def test_vector_rhs_warm_start(dtype, layout, guess):
    A, b = system(dtype)
    expected = torch.linalg.solve(A, b).unsqueeze(-1)
    x = None
    if guess == "zero":
        x = torch.zeros_like(b)
    elif guess == "nonzero":
        x = torch.tensor([.1, .2, -.3, .4], dtype=dtype)
    elif guess == "exact":
        x = expected.squeeze(-1).clone()
    elif guess == "column":
        x = torch.full_like(expected, .1)
    if layout == "coo":
        A = A.to_sparse_coo()
    elif layout == "csr":
        A = A.to_sparse_csr()
    elif layout == "bsr":
        A = A.to_sparse_bsr((2, 2))
    result = CG(tol=1e-6 if dtype == torch.float32 else 1e-12)(A, b, x=x)
    torch.testing.assert_close(result, expected)
    torch.testing.assert_close(A @ result, b.unsqueeze(-1))
    assert result.shape == (4, 1)
    assert result.dtype == dtype


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_preconditioned_vector_guess(dtype):
    A, b = system(dtype)
    expected = torch.linalg.solve(A, b).unsqueeze(-1)
    M = torch.diag(A.diag().reciprocal())
    result = CG()(A.to_sparse_csr(), b, x=torch.full_like(b, .2), M=M)
    torch.testing.assert_close(result, expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_noncontiguous_vector_guess(dtype):
    A, b = system(dtype)
    backing = torch.zeros(8, dtype=dtype)
    x = backing[::2]
    x.copy_(torch.tensor([.1, .2, -.3, .4], dtype=dtype))
    assert not x.is_contiguous()
    expected = torch.linalg.solve(A, b).unsqueeze(-1)
    result = CG()(A, b, x=x)
    torch.testing.assert_close(result, expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_zero_rhs_with_vector_guess(dtype):
    A, b = system(dtype)
    result = CG()(A, torch.zeros_like(b), x=torch.ones_like(b))
    torch.testing.assert_close(result, torch.zeros(4, 1, dtype=dtype))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_shared_iteration_batched_vector_guesses(dtype):
    A, b = system(dtype)
    A = A.expand(2, 4, 4)
    b = torch.stack((b, b))
    result = CG()(A, b, x=torch.full_like(b, .1))
    torch.testing.assert_close(result, torch.linalg.solve(A, b).unsqueeze(-1))
