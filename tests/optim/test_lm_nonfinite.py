import pytest
import torch
import pypose as pp


DEVICES = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])


class DomainModel(torch.nn.Module):
    def __init__(self, kind, device, dtype):
        super().__init__()
        self.kind = kind
        initial = 1.0 if kind == 'sqrt' else 0.0
        self.x = torch.nn.Parameter(torch.tensor([initial], device=device, dtype=dtype))

    def forward(self):
        return self.x.sqrt() if self.kind == 'sqrt' else self.x.exp()


class CountingSolver(torch.nn.Module):
    def __init__(self, invalid=False):
        super().__init__()
        self.calls = 0
        self.invalid = invalid
        self.solver = pp.optim.solver.Cholesky()

    def forward(self, A, b):
        self.calls += 1
        return torch.full_like(b, float('nan')) if self.invalid else self.solver(A, b)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_lm_retries_domain_overshoot(device, dtype):
    """A finite LM step outside sqrt's domain must not poison later Jacobians."""
    model = DomainModel('sqrt', device, dtype)
    target = model.x.new_tensor([0.1])
    optimizer = pp.optim.LM(model)
    loss = optimizer.step((), target)
    assert torch.isfinite(loss)
    assert 0 < model.x.item() < 1
    assert 0 < optimizer.reject_count <= optimizer.reject
    for _ in range(20):
        loss = optimizer.step((), target)
        assert torch.isfinite(loss)
        assert model.x.item() > 0
        if loss < 1e-10:
            break
    torch.testing.assert_close(model.x, model.x.new_tensor([0.01]), rtol=1e-3, atol=1e-5)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('kind', ['sqrt', 'exp'])
@pytest.mark.parametrize('reject', [0, 2])
@pytest.mark.parametrize('strategy', [pp.optim.strategy.Constant,
                                     pp.optim.strategy.Adaptive,
                                     pp.optim.strategy.TrustRegion])
def test_lm_nonfinite_retry_budget(device, kind, reject, strategy):
    """NaN/Inf trials restore the finite state even when no retries remain."""
    model = DomainModel(kind, device, torch.float64)
    target = model.x.new_tensor([0.1 if kind == 'sqrt' else 1000.0])
    solver = CountingSolver()
    optimizer = pp.optim.LM(model, solver=solver, strategy=strategy(), reject=reject)
    before = model.x.detach().clone()
    initial_loss = optimizer.model.loss((), target).detach().clone()
    for _ in range(2):
        calls = solver.calls
        loss = optimizer.step((), target)
        assert solver.calls - calls == reject + 1
        assert optimizer.reject_count == reject
        torch.testing.assert_close(model.x, before, rtol=0, atol=0)
        torch.testing.assert_close(loss, initial_loss, rtol=0, atol=0)
        torch.testing.assert_close(optimizer.model.loss((), target), loss, rtol=0, atol=0)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('liegroup', [False, True])
def test_lm_restores_parameters_after_nonfinite_update(device, liegroup):
    """An inverse update cannot recover a parameter already containing NaN."""
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.frozen = torch.nn.Parameter(torch.ones(1, device=device, dtype=torch.float64),
                                             requires_grad=False)
            value = torch.tensor([0.3, -0.2, 0.1, 0.2, -0.1, 0.4],
                                 device=device, dtype=torch.float64)
            self.x = pp.Parameter(pp.se3(value).Exp()) if liegroup else torch.nn.Parameter(value)

        def forward(self):
            value = self.x.Log().tensor() if liegroup else self.x
            return value * self.frozen

    model = Model()
    parameter = model.x
    before = model.x.detach().clone()
    frozen = model.frozen.detach().clone()
    optimizer = pp.optim.LM(model, solver=CountingSolver(invalid=True), reject=0)
    loss = optimizer.step(())
    assert torch.isfinite(loss)
    assert model.x is parameter
    torch.testing.assert_close(model.x, before, rtol=0, atol=0)
    torch.testing.assert_close(model.frozen, frozen, rtol=0, atol=0)
    torch.testing.assert_close(optimizer.model.loss((), None), loss, rtol=0, atol=0)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('reject', [0, 2])
def test_lm_finite_trial_behavior(device, reject):
    """Keep finite accepted updates and the existing finite retry-limit behavior."""
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.x = torch.nn.Parameter(torch.ones(1, device=device, dtype=torch.float64))

        def forward(self):
            return self.x

    model = Model()
    damping = 0.1
    optimizer = pp.optim.LM(model, strategy=pp.optim.strategy.Constant(damping), reject=reject)
    loss = optimizer.step(())
    expected = model.x.new_tensor([damping / (1 + damping)])
    torch.testing.assert_close(model.x, expected)
    torch.testing.assert_close(loss, expected.square().sum())
    assert optimizer.reject_count == 0

    class UphillSolver(torch.nn.Module):
        def forward(self, A, b):
            return torch.ones_like(b)

    model = Model()
    optimizer = pp.optim.LM(model, solver=UphillSolver(), reject=reject)
    loss = optimizer.step(())
    torch.testing.assert_close(model.x, model.x.new_tensor([2.0]))
    torch.testing.assert_close(loss, model.x.new_tensor(4.0))
    assert optimizer.reject_count == reject
