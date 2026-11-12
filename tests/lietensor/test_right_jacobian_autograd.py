import pytest
import torch
import pypose as pp


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("as_group", [False, True])
def test_right_jacobian_identity_gradient(dtype, as_group):
    vector = torch.zeros(2, 3, dtype=dtype, requires_grad=True)
    rotation = pp.so3(vector)
    if as_group:
        rotation = rotation.Exp()
    jacobian = pp.Jr(rotation)
    torch.testing.assert_close(jacobian, torch.eye(3, dtype=dtype).expand(2, 3, 3))
    gradient = torch.autograd.grad(jacobian[:, 0, 1].sum(), vector)[0]
    expected = torch.tensor([[0., 0., 0.5], [0., 0., 0.5]], dtype=dtype)
    torch.testing.assert_close(gradient, expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_right_jacobian_preserves_first_order_term_below_epsilon(dtype):
    angle = torch.finfo(dtype).eps / 4
    vector = torch.tensor([0., 0., angle], dtype=dtype, requires_grad=True)
    jacobian = pp.so3(vector).Jr()
    torch.testing.assert_close(jacobian[0, 1], torch.tensor(angle / 2, dtype=dtype),
                               rtol=1e-6, atol=0)
    gradient = torch.autograd.grad(jacobian[0, 1], vector)[0]
    torch.testing.assert_close(gradient, torch.tensor([0., 0., 0.5], dtype=dtype))


def test_right_jacobian_mixed_batch_gradcheck():
    vector = torch.tensor([[[0., 0., 0.], [0.3, -0.2, 0.4]],
                           [[-0.4, 0.1, 0.2], [0., 0., 0.]]],
                          dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda x: pp.so3(x).Jr(), (vector,),
                                    eps=1e-4, atol=1e-7, rtol=1e-5)


def test_right_jacobian_ordinary_angles_match_closed_form():
    vector = torch.tensor([[0.3, -0.2, 0.4], [-0.4, 0.1, 0.2]], dtype=torch.float64)
    angle = vector.norm(dim=-1, keepdim=True).unsqueeze(-1)
    skew = pp.vec2skew(vector)
    expected = (torch.eye(3, dtype=vector.dtype)
                - (1 - angle.cos()) / angle.square() * skew
                + (angle - angle.sin()) / angle.pow(3) * (skew @ skew))
    torch.testing.assert_close(pp.so3(vector).Jr(), expected)
