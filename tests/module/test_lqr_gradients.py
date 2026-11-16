import pytest
import torch
import pypose as pp


def dense_lqr(A, B, c, Q, p, x_init):
    batch, horizon, nsc = p.shape
    ns, nc = A.shape[-1], B.shape[-1]
    width = horizon * nc
    eye = torch.eye(width, dtype=p.dtype, device=p.device)
    offset = x_init
    mapping = p.new_zeros(batch, ns, width)
    hessian = p.new_zeros(batch, width, width)
    gradient = p.new_zeros(batch, width)

    for t in range(horizon):
        control = eye[t * nc:(t + 1) * nc].expand(batch, -1, -1)
        stage_mapping = torch.cat((mapping, control), dim=-2)
        stage_offset = torch.cat((offset, p.new_zeros(batch, nc)), dim=-1)
        hessian = hessian + stage_mapping.mT @ Q[:, t] @ stage_mapping
        gradient = gradient + (
            stage_mapping.mT
            @ (Q[:, t] @ stage_offset.unsqueeze(-1) + p[:, t].unsqueeze(-1))
        ).squeeze(-1)
        offset = (A @ offset.unsqueeze(-1)).squeeze(-1) + c
        mapping = A @ mapping + B @ control

    u = torch.linalg.solve(hessian, -gradient.unsqueeze(-1)).reshape(batch, horizon, nc)
    states = [x_init]
    for t in range(horizon):
        states.append(
            (A @ states[-1].unsqueeze(-1) + B @ u[:, t].unsqueeze(-1)).squeeze(-1) + c
        )
    x = torch.stack(states, dim=1)
    trajectory = torch.cat((x[:, :-1], u), dim=-1)
    cost = (
        0.5 * torch.einsum("bti,btij,btj->bt", trajectory, Q, trajectory)
        + (trajectory * p).sum(dim=-1)
    ).sum(dim=-1)
    return x, u, cost


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
@pytest.mark.parametrize("warm_start", (False, True))
@pytest.mark.parametrize("batch,horizon", ((1, 1), (1, 8), (3, 1), (3, 8)))
def test_lqr_outputs_and_gradients_match_condensed_problem(dtype, warm_start, batch, horizon):
    generator = torch.Generator().manual_seed(41)
    ns, nc = 3, 2
    nsc = ns + nc

    def randn(*shape):
        return torch.randn(*shape, generator=generator, dtype=dtype)

    A = 0.7 * torch.eye(ns, dtype=dtype).expand(batch, -1, -1) + 0.02 * randn(batch, ns, ns)
    B = 0.2 * randn(batch, ns, nc)
    c = 0.1 * randn(batch, ns)
    factor = 0.1 * randn(batch, horizon, nsc, nsc)
    Q = factor.mT @ factor + torch.eye(nsc, dtype=dtype)
    initial = randn(batch, ns)
    linear = 0.1 * randn(batch, horizon, nsc)
    nominal = 0.2 * randn(batch, horizon, nc) if warm_start else None

    x_init = initial.clone().requires_grad_()
    p = linear.clone().requires_grad_()
    system = pp.module.LTI(
        A, B, torch.eye(ns, dtype=dtype).expand(batch, -1, -1),
        torch.zeros(batch, ns, nc, dtype=dtype), c, torch.zeros_like(c)
    )
    actual = pp.module.LQR(system, Q, p, horizon)(x_init, u_traj=nominal)
    actual_loss = sum((i + 1) * value.square().sum() for i, value in enumerate(actual))
    actual_grad = torch.autograd.grad(actual_loss, (x_init, p))

    reference_x = initial.clone().requires_grad_()
    reference_p = linear.clone().requires_grad_()
    expected = dense_lqr(A, B, c, Q, reference_p, reference_x)
    expected_loss = sum((i + 1) * value.square().sum() for i, value in enumerate(expected))
    expected_grad = torch.autograd.grad(expected_loss, (reference_x, reference_p))

    rtol, atol = (2e-4, 2e-5) if dtype == torch.float32 else (1e-10, 1e-11)
    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result, reference, rtol=rtol, atol=atol)
    for result, reference in zip(actual_grad, expected_grad):
        torch.testing.assert_close(result, reference, rtol=rtol, atol=atol)
