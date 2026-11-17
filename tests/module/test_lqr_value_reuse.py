import pytest
import torch
import pypose as pp


def lqr_problem(batch, horizon, dtype, gradients=False):
    generator = torch.Generator().manual_seed(124)

    def randn(*shape):
        return torch.randn(*shape, generator=generator, dtype=dtype)

    ns, nc = 3, 2
    factor = (0.3 * randn(batch, horizon, ns + nc, ns + nc)).requires_grad_(gradients)
    Q = factor.mT @ factor + 0.5 * torch.eye(ns + nc, dtype=dtype)
    p = randn(batch, horizon, ns + nc).requires_grad_(gradients)
    x_init = randn(batch, ns).requires_grad_(gradients)
    A = 0.4 * torch.eye(ns, dtype=dtype) + 0.05 * randn(batch, ns, ns)
    B = 0.3 * randn(batch, ns, nc)
    c = 0.1 * randn(batch, ns)
    C = torch.eye(ns, dtype=dtype).expand(batch, ns, ns)
    D = torch.zeros(batch, ns, nc, dtype=dtype)
    system = pp.module.LTI(A, B, C, D, c)
    warm_start = 0.2 * randn(batch, horizon, nc)
    return system, Q, p, x_init, warm_start, factor


def dense_lqr(A, B, c, Q, p, x_init):
    """Eliminate the dynamics and solve one dense QP over all controls."""
    batch, horizon, _ = p.shape
    ns, nc = B.shape[-2:]
    nu = horizon * nc
    selectors = torch.eye(nu, dtype=Q.dtype).reshape(horizon, nc, nu)
    state_map = Q.new_zeros(batch, ns, nu)
    state_offset = x_init
    hessian = Q.new_zeros(batch, nu, nu)
    linear = Q.new_zeros(batch, nu)

    for t in range(horizon):
        selector = selectors[t].expand(batch, nc, nu)
        mapping = torch.cat((state_map, selector), dim=-2)
        offset = torch.cat((state_offset, Q.new_zeros(batch, nc)), dim=-1)
        hessian = hessian + mapping.mT @ Q[:, t] @ mapping
        residual = (Q[:, t] @ offset.unsqueeze(-1)).squeeze(-1) + p[:, t]
        linear = linear + (mapping.mT @ residual.unsqueeze(-1)).squeeze(-1)
        state_map = A @ state_map + B @ selector
        state_offset = (A @ state_offset.unsqueeze(-1)).squeeze(-1) + c

    controls = -torch.linalg.solve(hessian, linear.unsqueeze(-1))
    controls = controls.reshape(batch, horizon, nc)
    states = [x_init]
    cost = Q.new_zeros(batch)
    for t in range(horizon):
        tau = torch.cat((states[-1], controls[:, t]), dim=-1)
        cost = cost + 0.5 * torch.einsum('bi,bij,bj->b', tau, Q[:, t], tau)
        cost = cost + (tau * p[:, t]).sum(-1)
        state = A @ states[-1].unsqueeze(-1) + B @ controls[:, t].unsqueeze(-1)
        states.append(state.squeeze(-1) + c)
    return torch.stack(states, dim=1), controls, cost


@pytest.mark.parametrize('batch', [1, 3])
@pytest.mark.parametrize('horizon', [1, 2, 5])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('warm_start', [False, True])
def test_lqr_matches_dense_qp(batch, horizon, dtype, warm_start):
    system, Q, p, x_init, nominal, _ = lqr_problem(batch, horizon, dtype)
    result = pp.module.LQR(system, Q, p, horizon)(
        x_init, u_traj=nominal if warm_start else None)
    expected = dense_lqr(system.A, system.B, system.c1, Q, p, x_init)
    tolerance = 2e-5 if dtype == torch.float32 else 2e-11
    for actual, reference in zip(result, expected):
        torch.testing.assert_close(actual, reference, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize('horizon', [1, 2, 5])
def test_lqr_gradients_match_dense_qp(horizon):
    system, Q, p, x_init, nominal, factor = lqr_problem(
        2, horizon, torch.float64, gradients=True)
    result = pp.module.LQR(system, Q, p, horizon)(x_init, u_traj=nominal)
    expected = dense_lqr(system.A, system.B, system.c1, Q, p, x_init)
    inputs = (factor, p, x_init)
    loss = sum((i + 1) * output.square().sum() for i, output in enumerate(result))
    reference_loss = sum((i + 1) * output.square().sum()
                         for i, output in enumerate(expected))
    gradients = torch.autograd.grad(loss, inputs, retain_graph=True)
    expected_gradients = torch.autograd.grad(reference_loss, inputs)
    for actual, reference in zip(gradients, expected_gradients):
        torch.testing.assert_close(actual, reference, atol=2e-9, rtol=2e-8)
