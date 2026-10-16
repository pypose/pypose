import pytest
import torch
import pypose as pp
from pypose.optim.functional import modjac


@pytest.mark.parametrize("batch_shape", [(), (2,), (2, 3)])
def test_se3_translation_autograd(batch_shape):
    dtype = torch.float64
    xi1 = pp.randn_se3(*batch_shape, dtype=dtype, requires_grad=True)
    xi2 = xi1.detach().clone().requires_grad_(True)

    out1 = xi1.Exp().translation()
    out2 = xi2.Exp().Act(torch.zeros(3, dtype=dtype))

    grad_output = torch.randn_like(out1)
    g1 = torch.autograd.grad(out1, xi1, grad_outputs=grad_output)[0]
    g2 = torch.autograd.grad(out2, xi2, grad_outputs=grad_output)[0]

    torch.testing.assert_close(g1, g2)


@pytest.mark.parametrize("batch_shape", [(), (2,), (2, 3)])
def test_sim3_translation_autograd(batch_shape):
    dtype = torch.float64
    xi1 = pp.randn_sim3(*batch_shape, dtype=dtype, requires_grad=True)
    xi2 = xi1.detach().clone().requires_grad_(True)

    out1 = xi1.Exp().translation()
    out2 = xi2.Exp().Act(torch.zeros(3, dtype=dtype))

    grad_output = torch.randn_like(out1)
    g1 = torch.autograd.grad(out1, xi1, grad_outputs=grad_output)[0]
    g2 = torch.autograd.grad(out2, xi2, grad_outputs=grad_output)[0]

    torch.testing.assert_close(g1, g2)


def test_se3_translation_finite_difference():
    dtype = torch.float64

    class Probe(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.xi = pp.Parameter(
                pp.se3(torch.tensor([0.2, 0.3, 0.1, 0.2, 0.3, 0.1], dtype=dtype))
            )

        def forward(self):
            return self.xi.Exp().translation()

    def finite_difference_jacobian(module, eps=1e-7):
        p = module.xi
        J = torch.zeros(3, 6, dtype=dtype)
        for i in range(6):
            p.data.view(-1)[i] += eps
            rp = module().clone()

            p.data.view(-1)[i] -= 2 * eps
            rm = module().clone()

            p.data.view(-1)[i] += eps
            J[:, i] = (rp - rm) / (2 * eps)
        return J

    m = Probe()
    J_auto = modjac(m, input=(), flatten=False, vectorize=True)
    J_auto = (J_auto[0] if isinstance(J_auto, tuple) else J_auto).reshape(3, 6)
    J_fd = finite_difference_jacobian(Probe())

    torch.testing.assert_close(J_auto, J_fd, rtol=1e-5, atol=1e-5)


def test_se3_translation_gradcheck():
    def func(xi):
        return pp.se3(xi).Exp().translation()

    xi = torch.randn(2, 6, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(func, (xi,))


def test_translation_empty_tensor():
    x = pp.randn_SE3(0, requires_grad=True)
    t = x.translation()
    assert t.shape == (0, 3)
    loss = t.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == (0, 7)
