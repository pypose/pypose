import pytest
import torch
import pypose as pp


class TimedModel(pp.module.NLS):
    def __init__(self, none_fallback=False):
        super().__init__()
        self.none_fallback = none_fallback

    def time_value(self, state, t):
        if t is None and self.none_fallback:
            return state.new_tensor(0.)
        return t.to(state.dtype)

    def state_transition(self, state, input, t=None):
        t = self.time_value(state, t)
        return (1 + 0.1 * t) * state + input + t

    def observation(self, state, input, t=None):
        t = self.time_value(state, t)
        return (1 + 0.05 * t) * state + 0.2 * t


def reference(x, y, u, P, Q, R, time):
    a, h = 1 + 0.1 * time, 1 + 0.05 * time
    predicted = a * x + u + time
    covariance = a * a * P + Q
    gain = covariance * h / (h * h * covariance + R)
    return predicted + gain.squeeze(-1) * (y - h * predicted - 0.2 * time), (
        1 - gain * h
    ) * covariance


def inputs(dtype):
    return [
        torch.tensor(value, dtype=dtype)
        for value in ([0.7], [4.8], [0.2], [[0.6]], [[0.1]], [[0.3]])
    ]


@pytest.mark.parametrize("Filter", [pp.module.EKF, pp.module.UKF])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("time", [0, 3])
@pytest.mark.parametrize("none_fallback", [False, True])
def test_omitted_time_uses_current_clock(Filter, dtype, time, none_fallback):
    model = TimedModel(none_fallback).reset(time)
    filt = Filter(model)
    args = inputs(dtype)
    actual = filt(*args)
    expected = reference(*args, time)
    for result, target in zip(actual, expected):
        torch.testing.assert_close(result, target)
    assert model.systime.item() == time


@pytest.mark.parametrize("Filter", [pp.module.EKF, pp.module.UKF])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("none_fallback", [False, True])
def test_explicit_time_still_overrides_clock(Filter, dtype, none_fallback):
    model = TimedModel(none_fallback).reset(9)
    filt = Filter(model)
    args = inputs(dtype)
    actual = filt(*args, t=torch.tensor(2))
    expected = reference(*args, 2)
    for result, target in zip(actual, expected):
        torch.testing.assert_close(result, target)
    assert model.systime.item() == 9


@pytest.mark.parametrize("Filter", [pp.module.EKF, pp.module.UKF])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_repeated_updates_follow_clock_changes(Filter, dtype):
    model = TimedModel().reset(1)
    filt = Filter(model)
    x, y, u, P, Q, R = inputs(dtype)
    expected_x, expected_P = x.clone(), P.clone()
    for time in (1, 4, 2):
        model.reset(time)
        x, P = filt(x, y, u, P, Q, R)
        expected_x, expected_P = reference(
            expected_x, y, u, expected_P, Q, R, time
        )
        torch.testing.assert_close(x, expected_x)
        torch.testing.assert_close(P, expected_P)


class ClockProduct(pp.module.NLS):
    def state_transition(self, state, input, t):
        return state * t + input

    def observation(self, state, input, t):
        return state


@pytest.mark.parametrize("Filter", [pp.module.EKF, pp.module.UKF])
def test_later_clock_update_preserves_prediction_backward(Filter):
    model = ClockProduct().reset(3)
    filt = Filter(model)
    args = inputs(torch.float64)
    args[0].requires_grad_()
    args[2].requires_grad_()
    actual = filt(*args)
    model.reset(8)
    gradients = torch.autograd.grad(actual[0].sum(), (args[0], args[2]))

    control = Filter(ClockProduct().reset(8))
    other = inputs(torch.float64)
    other[0].requires_grad_()
    other[2].requires_grad_()
    expected = control(*other, t=torch.tensor(3))
    expected_gradients = torch.autograd.grad(
        expected[0].sum(), (other[0], other[2])
    )
    for result, target in zip(actual, expected):
        torch.testing.assert_close(result, target)
    for result, target in zip(gradients, expected_gradients):
        torch.testing.assert_close(result, target)


class TimeIndependent(pp.module.NLS):
    def state_transition(self, state, input, t=None):
        return state + input

    def observation(self, state, input, t=None):
        return state


@pytest.mark.parametrize("Filter", [pp.module.EKF, pp.module.UKF])
def test_time_independent_model_retains_result(Filter):
    args = inputs(torch.float64)
    first = Filter(TimeIndependent().reset(0))(*args)
    later = Filter(TimeIndependent().reset(19))(*args)
    for result, target in zip(first, later):
        torch.testing.assert_close(result, target)

