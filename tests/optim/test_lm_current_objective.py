import pytest
import torch
import pypose as pp


class ScalarFit(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.value = torch.nn.Parameter(torch.ones(1, dtype=torch.float64))

    def forward(self, input):
        return self.value * input


@pytest.mark.parametrize("strategy_type", [
    pp.optim.strategy.Constant, pp.optim.strategy.Adaptive, pp.optim.strategy.TrustRegion,
])
@pytest.mark.parametrize("change", ["target", "input", "parameter"])
def test_lm_compares_trials_with_current_objective(strategy_type, change):
    model = ScalarFit()
    optimizer = pp.optim.LM(model, strategy=strategy_type())
    inputs = torch.ones(1, dtype=torch.float64)
    target = torch.ones(1, dtype=torch.float64)
    torch.testing.assert_close(optimizer.step(inputs, target), torch.zeros_like(target).sum())

    if change == "target":
        target = torch.full_like(target, 10)
    elif change == "input":
        inputs = torch.full_like(inputs, 2)
    else:
        with torch.no_grad():
            model.value.fill_(4)

    current_loss = (model(inputs) - target).square().sum().detach()
    loss = optimizer.step(inputs, target)
    torch.testing.assert_close(optimizer.last, current_loss)
    assert optimizer.reject_count == 0
    assert loss < current_loss * 1e-8
    torch.testing.assert_close(model(inputs), target, atol=1e-4, rtol=1e-4)
