import numpy as np
import pytest
import torch
import pypose as pp


def covariance_reference(a, b, weight=None, noise=0):
    products = a.unsqueeze(-1) * b.unsqueeze(-2)
    covariance = products.mean(0) if weight is None else (weight.unsqueeze(-1) * products).sum(0)
    return covariance + noise


@pytest.mark.parametrize("kind", ["PF", "UKF"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("batch", [(), (3,), (2, 3)])
@pytest.mark.parametrize("samples", [1, 7])
def test_covariance_matches_independent_numpy_reduction(kind, dtype, batch, samples):
    generator = torch.Generator().manual_seed(41)
    a = torch.randn(samples, *batch, 4, dtype=dtype, generator=generator)
    b = torch.randn(samples, *batch, 3, dtype=dtype, generator=generator)
    a = a.movedim(0, -1).contiguous().movedim(-1, 0)
    noise = torch.randn(*batch, 4, 3, dtype=dtype, generator=generator) / 20
    weight = torch.full((samples, *batch, 1), 1 / samples, dtype=dtype)
    if samples > 1:
        weight[0] = -0.5
        weight[1:] = 1.5 / (samples - 1)
    filter = getattr(pp.module, kind)(pp.module.NLS())

    actual = filter.compute_cov(a, b, noise) if kind == "PF" else filter.compute_cov(a, b, weight, noise)

    # Sum each cross-covariance entry independently in NumPy float64.
    an, bn = a.double().numpy(), b.double().numpy()
    wn = np.full((samples, *batch), 1 / samples) if kind == "PF" else weight[..., 0].double().numpy()
    expected = np.empty((*batch, 4, 3))
    for row in range(4):
        for column in range(3):
            expected[..., row, column] = np.sum(wn * an[..., row] * bn[..., column], axis=0)
    expected += noise.double().numpy()
    rtol, atol = (2e-5, 2e-6) if dtype == torch.float32 else (1e-12, 1e-12)
    torch.testing.assert_close(actual, torch.as_tensor(expected, dtype=dtype), rtol=rtol, atol=atol)


@pytest.mark.parametrize("kind", ["PF", "UKF"])
@pytest.mark.parametrize("sample_broadcast", [False, True])
def test_existing_sample_and_batch_broadcasts(kind, sample_broadcast):
    a = torch.arange(8. if sample_broadcast else 28.).reshape(1 if sample_broadcast else 7, -1, 4)
    b = torch.arange(42. if sample_broadcast else 63.).reshape(7, -1, 3)
    weight = torch.full((7, 1, 1), 1 / 7)
    filter = getattr(pp.module, kind)(pp.module.NLS())
    actual = filter.compute_cov(a, b) if kind == "PF" else filter.compute_cov(a, b, weight)
    torch.testing.assert_close(actual, covariance_reference(a, b, None if kind == "PF" else weight))


@pytest.mark.parametrize("kind", ["PF", "UKF"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_reduction_retains_existing_range(kind, dtype):
    a = torch.full((10000, 2), 10., dtype=dtype)
    b = torch.full((10000, 3), 10., dtype=dtype)
    weight = torch.full((10000, 1), 1 / 10000, dtype=dtype)
    filter = getattr(pp.module, kind)(pp.module.NLS())
    actual = filter.compute_cov(a, b) if kind == "PF" else filter.compute_cov(a, b, weight)
    expected = covariance_reference(a, b, None if kind == "PF" else weight)
    assert actual.isfinite().all()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["PF", "UKF"])
def test_empty_sample_reduction_retains_existing_result(kind):
    a, b = torch.empty(0, 2, 4), torch.empty(0, 2, 3)
    weight = torch.empty(0, 2, 1)
    filter = getattr(pp.module, kind)(pp.module.NLS())
    actual = filter.compute_cov(a, b) if kind == "PF" else filter.compute_cov(a, b, weight)
    torch.testing.assert_close(actual, covariance_reference(a, b, None if kind == "PF" else weight), equal_nan=True)


@pytest.mark.parametrize("kind", ["PF", "UKF"])
def test_covariance_gradients_and_second_derivatives(kind):
    generator = torch.Generator().manual_seed(19)
    a = torch.randn(3, 2, 2, dtype=torch.float64, generator=generator, requires_grad=True)
    b = torch.randn(3, 2, 3, dtype=torch.float64, generator=generator, requires_grad=True)
    weight = torch.randn(3, 2, 1, dtype=torch.float64, generator=generator, requires_grad=True)
    noise = torch.randn(2, 2, 3, dtype=torch.float64, generator=generator, requires_grad=True)
    filter = getattr(pp.module, kind)(pp.module.NLS())
    if kind == "PF":
        function = lambda a, b, noise: filter.compute_cov(a, b, noise)
        inputs = (a, b, noise)
    else:
        function = lambda a, b, noise, weight: filter.compute_cov(a, b, weight, noise)
        inputs = (a, b, noise, weight)
    actual = torch.autograd.grad(function(*inputs).square().sum(), inputs)
    reference = torch.autograd.grad(covariance_reference(a, b, weight if kind == "UKF" else None, noise).square().sum(), inputs)
    for computed, expected in zip(actual, reference):
        torch.testing.assert_close(computed, expected)
    assert torch.autograd.gradcheck(function, inputs)
    assert torch.autograd.gradgradcheck(function, inputs)


class LinearSystem(pp.module.NLS):
    def __init__(self, dtype):
        super().__init__()
        self.register_buffer("transition", 0.9 * torch.eye(4, dtype=dtype))
        self.register_buffer("measurement", torch.tensor([[1., 0., 0.2, 0.], [0., 1., 0., -0.1]], dtype=dtype))

    def state_transition(self, state, input, t=None):
        return (self.transition @ state.unsqueeze(-1)).squeeze(-1) + input

    def observation(self, state, input, t=None):
        return (self.measurement @ state.unsqueeze(-1)).squeeze(-1)

    def forward(self, state, input):
        state = self.state_transition(state, input)
        return state, self.observation(state, input)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("batch", [(), (2,), (2, 3)])
def test_public_ukf_matches_linear_gaussian_reference(dtype, batch):
    model = LinearSystem(dtype)
    filter = pp.module.UKF(model)
    x = torch.full((*batch, 4), 0.2, dtype=dtype)
    u = torch.full_like(x, 0.03)
    y = torch.full((*batch, 2), 0.4, dtype=dtype)
    P, Q, R = 0.2 * torch.eye(4, dtype=dtype), 0.03 * torch.eye(4, dtype=dtype), 0.1 * torch.eye(2, dtype=dtype)
    for _ in range(3):
        prior = model.transition @ P @ model.transition.mT + Q
        predicted = model.state_transition(x, u)
        covariance = model.measurement @ prior @ model.measurement.mT + R
        gain = prior @ model.measurement.mT @ torch.linalg.inv(covariance)
        expected_x = predicted + (gain @ (y - model.observation(predicted, u)).unsqueeze(-1)).squeeze(-1)
        expected_P = prior - gain @ covariance @ gain.mT
        x, P = filter(x, y, u, P, Q, R)
        torch.testing.assert_close(x, expected_x)
        torch.testing.assert_close(P, expected_P.expand_as(P))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("batch", [(), (2, 3)])
def test_public_pf_preserves_updates_with_same_random_draws(dtype, batch):
    class OriginalCovariancePF(pp.module.PF):
        def compute_cov(self, a, b, Q=0):
            return covariance_reference(a, b, noise=Q)

    Q, R = 0.03 * torch.eye(4, dtype=dtype), 0.1 * torch.eye(2, dtype=dtype)
    outputs = []
    for filter_type in (OriginalCovariancePF, pp.module.PF):
        filter = filter_type(LinearSystem(dtype), Q, R, particles=128)
        x = torch.full((*batch, 4), 0.2, dtype=dtype)
        P = 0.2 * torch.eye(4, dtype=dtype)
        u, y = torch.full_like(x, 0.03), torch.full((*batch, 2), 0.4, dtype=dtype)
        with torch.random.fork_rng():
            torch.manual_seed(11)
            steps = []
            for _ in range(3):
                x, P = filter(x, y, u, P)
                steps.append((x, P))
        outputs.append(steps)
    for expected, actual in zip(*outputs):
        for expected_value, actual_value in zip(expected, actual):
            torch.testing.assert_close(actual_value, expected_value)
