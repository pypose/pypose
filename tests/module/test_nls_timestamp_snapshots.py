import pytest
import torch
import pypose as pp


class TimeVaryingSystem(pp.module.NLS):
    def __init__(self, use_clock=False):
        super().__init__()
        self.use_clock = use_clock

    def state_transition(self, state, input, t):
        t = self.systime if self.use_clock else t
        return t * state.square() + input

    def observation(self, state, input, t):
        t = self.systime if self.use_clock else t
        return t * state + input.square()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("batch", [(), (2,), (2, 3)])
@pytest.mark.parametrize("module_call", [False, True])
@pytest.mark.parametrize("use_clock", [False, True])
def test_delayed_backward_uses_each_calls_timestamp(dtype, batch, module_call, use_clock):
    model = TimeVaryingSystem(use_clock).reset(2)
    state = torch.full((*batch, 2), 0.3, dtype=dtype, requires_grad=True)
    control = torch.full_like(state, 0.2, requires_grad=True)
    loss = 0
    timestamps = []
    for step in range(3):
        timestamp = 2 + step if module_call else 2
        timestamps.append(timestamp)
        output, observed = model(state, control) if module_call else model.forward(state, control)
        torch.testing.assert_close(output, timestamp * state.square() + control)
        torch.testing.assert_close(observed, timestamp * state + control.square())
        loss = loss + output.sum() + observed.sum()

    loss.backward()

    torch.testing.assert_close(state.grad, (2 * state.detach() + 1) * sum(timestamps))
    torch.testing.assert_close(control.grad, 3 * (1 + 2 * control.detach()))
    assert model.systime.item() == (5 if module_call else 2)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_clock_remains_a_serialized_integer_buffer(dtype):
    model = TimeVaryingSystem().to(dtype=dtype).reset(2)
    state = torch.tensor([0.3], dtype=dtype)
    control = torch.tensor([0.2], dtype=dtype)
    model(state, control)
    assert dict(model.named_buffers())["_t"] is model.systime
    assert model.systime.dtype == torch.int64
    saved = {name: value.clone() for name, value in model.state_dict().items()}
    restored = TimeVaryingSystem().to(dtype=dtype)
    restored.load_state_dict(saved)
    expected = model(state, control)
    actual = restored(state, control)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b)
    assert restored.systime.item() == model.systime.item() == 4


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("time_source, change", [
    ("clock", "forward"), ("clock", "reset"), ("clock", "assign"),
    ("explicit", "forward"), ("explicit", "external"),
    ("view", "external"), ("explicit", "reset"), ("view", "forward"),
])
def test_saved_linearization_time_is_independent_of_later_clock_changes(dtype, time_source, change):
    model = TimeVaryingSystem().reset(2)
    state = torch.tensor([3., -2.], dtype=dtype)
    control = torch.tensor([0.5, -0.25], dtype=dtype)
    storage = torch.tensor([9., 2., 8.], dtype=dtype)
    timestamp = storage[1] if time_source == "view" else torch.tensor(2., dtype=dtype)
    model.set_refpoint(state, control, None if time_source == "clock" else timestamp)

    if change == "forward":
        model(torch.ones_like(state), torch.zeros_like(control))
    elif change == "reset":
        model.reset(7)
    elif change == "assign":
        model.systime = 9
    else:
        timestamp.fill_(6)

    torch.testing.assert_close(model.A, torch.diag(4 * state))
    torch.testing.assert_close(model.B, torch.eye(2, dtype=dtype))
    torch.testing.assert_close(model.C, 2 * torch.eye(2, dtype=dtype))
    torch.testing.assert_close(model.D, torch.diag(2 * control))
    torch.testing.assert_close(model.c1, -2 * state.square())
    torch.testing.assert_close(model.c2, -control.square())
    assert model._ref_t.item() == 2


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("explicit_time", [False, True])
def test_reference_backward_survives_clock_or_timestamp_mutation(dtype, explicit_time):
    model = TimeVaryingSystem().reset(2)
    state = torch.tensor([3., -2.], dtype=dtype, requires_grad=True)
    control = torch.tensor([0.5, -0.25], dtype=dtype, requires_grad=True)
    timestamp = torch.tensor(2., dtype=dtype, requires_grad=True)
    model.set_refpoint(state, control, timestamp if explicit_time else None)
    with torch.no_grad():
        timestamp.add_(4)
    model.reset(9)

    (model._ref_f.sum() + model._ref_g.sum()).backward()

    torch.testing.assert_close(state.grad, 4 * state.detach() + 2)
    torch.testing.assert_close(control.grad, 1 + 2 * control.detach())
    if explicit_time:
        torch.testing.assert_close(timestamp.grad, (state.detach().square() + state.detach()).sum())


def test_explicit_time_keeps_first_and_second_derivatives():
    def function(state, control, timestamp):
        model = TimeVaryingSystem()
        model.jacargs["create_graph"] = True
        model.set_refpoint(state, control, timestamp)
        return torch.cat((model._ref_f, model._ref_g, model.A.flatten(), model.C.flatten()))

    inputs = (torch.tensor([0.3, -0.2], dtype=torch.float64, requires_grad=True),
              torch.tensor([0.2, 0.1], dtype=torch.float64, requires_grad=True),
              torch.tensor(2., dtype=torch.float64, requires_grad=True))
    assert torch.autograd.gradcheck(function, inputs)
    assert torch.autograd.gradgradcheck(function, inputs)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_real_parameter_training_matches_explicit_time_rollout(dtype):
    class LearnedSystem(pp.module.NLS):
        def __init__(self):
            super().__init__()
            self.gain = torch.nn.Parameter(torch.tensor(0.1, dtype=dtype))

        def state_transition(self, state, input, t):
            return self.gain * t * state + input

        def observation(self, state, input, t):
            return t * state

    model = LearnedSystem()
    reference_gain = torch.nn.Parameter(model.gain.detach().clone())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    reference_optimizer = torch.optim.SGD([reference_gain], lr=0.05)
    for _ in range(3):
        model.reset(2)
        optimizer.zero_grad()
        reference_optimizer.zero_grad()
        state = torch.tensor([0.2], dtype=dtype)
        reference = state.clone()
        control = torch.tensor([0.03], dtype=dtype)
        loss, reference_loss = 0, 0
        for timestamp in range(2, 6):
            state, observation = model(state, control)
            reference_observation = timestamp * reference
            reference = reference_gain * timestamp * reference + control
            loss = loss + state.square().sum() + observation.square().sum()
            reference_loss = reference_loss + reference.square().sum() + reference_observation.square().sum()
        loss.backward()
        reference_loss.backward()
        torch.testing.assert_close(model.gain.grad, reference_gain.grad)
        optimizer.step()
        reference_optimizer.step()
        torch.testing.assert_close(model.gain, reference_gain)
        assert model.systime.item() == 6
