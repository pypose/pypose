import pytest
import torch
import pypose as pp


class NoObservation(pp.module.NLS):
    def state_transition(self, state, input, t=None):
        return state

    def observation(self, state, input, t=None):
        return torch.zeros_like(state[..., :1])


class IdentityObservation(pp.module.NLS):
    def state_transition(self, state, input, t=None):
        return state

    def observation(self, state, input, t=None):
        return state


def prior(batch, shared, dtype):
    values = torch.arange(torch.Size(batch).numel(), dtype=dtype).reshape(batch)
    x = torch.stack((0.2 + 0.3 * values, -0.1 - 0.2 * values), dim=-1)
    covariance = torch.tensor([[1., 0.3], [0.3, 0.5]], dtype=dtype)
    if not shared:
        covariance = (1 + 0.2 * values[..., None, None]) * covariance
    return x, covariance


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("batch, shared", [((), True), ((3,), True), ((2, 3), False)])
def test_pf_no_information_preserves_prior_covariance(dtype, batch, shared):
    x, P = prior(batch, shared, dtype)
    pf = pp.module.PF(NoObservation(), particles=100000)
    with torch.random.fork_rng():
        torch.manual_seed(17)
        mean, covariance = pf(
            x, torch.zeros(*batch, 1, dtype=dtype), torch.zeros_like(x),
            P, torch.zeros_like(P), torch.ones(1, 1, dtype=dtype))

    # A constant observation has uniform likelihood. Real Gaussian sampling
    # and multinomial resampling should preserve the supplied prior moments.
    torch.testing.assert_close(mean, x, rtol=0, atol=0.03)
    torch.testing.assert_close(covariance, P.expand_as(covariance), rtol=0.025, atol=0.02)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("batch, shared", [((), True), ((3,), True), ((2, 3), False)])
def test_pf_matches_linear_gaussian_posterior(dtype, batch, shared):
    x, P = prior(batch, shared, dtype)
    y = x + torch.tensor([0.8, -0.4], dtype=dtype)
    R = torch.eye(2, dtype=dtype) * 0.8
    gain = torch.linalg.solve(P + R, P.mT).mT
    expected_mean = x + (gain @ (y - x).unsqueeze(-1)).squeeze(-1)
    expected_covariance = P - gain @ P
    pf = pp.module.PF(IdentityObservation(), particles=100000)

    with torch.random.fork_rng():
        torch.manual_seed(23)
        mean, covariance = pf(x, y, torch.zeros_like(x), P, torch.zeros_like(P), R)

    torch.testing.assert_close(mean, expected_mean, rtol=0, atol=0.025)
    torch.testing.assert_close(
        covariance, expected_covariance.expand_as(covariance), rtol=0.025, atol=0.015)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_pf_repeated_no_information_updates_do_not_inflate_uncertainty(dtype):
    x, P = prior((), True, dtype)
    expected_x, expected_P = x.clone(), P.clone()
    pf = pp.module.PF(NoObservation(), particles=100000)

    with torch.random.fork_rng():
        torch.manual_seed(31)
        for _ in range(3):
            x, P = pf(x, torch.zeros(1, dtype=dtype), torch.zeros_like(x),
                      P, torch.zeros_like(P), torch.ones(1, 1, dtype=dtype))

    torch.testing.assert_close(x, expected_x, rtol=0, atol=0.04)
    torch.testing.assert_close(P, expected_P, rtol=0.04, atol=0.025)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_pf_one_dimensional_prior_remains_unchanged(dtype):
    x = torch.tensor([0.2], dtype=dtype)
    P = torch.tensor([[0.5]], dtype=dtype)
    pf = pp.module.PF(NoObservation(), particles=100000)
    with torch.random.fork_rng():
        torch.manual_seed(17)
        mean, covariance = pf(x, torch.zeros(1, dtype=dtype), torch.zeros_like(x),
                              P, torch.zeros_like(P), torch.ones(1, 1, dtype=dtype))
    torch.testing.assert_close(mean, x, rtol=0, atol=0.025)
    torch.testing.assert_close(covariance, P, rtol=0.025, atol=0.015)
