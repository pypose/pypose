from unittest.mock import Mock

from pypose.optim.optimizer import _Optimizer
from pypose.optim.scheduler import StopOnPlateau


def test_checkpoint_keeps_destination_scheduler_bindings():
    source = StopOnPlateau(Mock(spec=_Optimizer), steps=3)
    target_optimizer = Mock(spec=_Optimizer)
    target = StopOnPlateau(target_optimizer, steps=5)
    state = source.state_dict()
    assert "continual" not in state
    assert "optimizer" not in state
    state.update(continual=source.continual, optimizer=source.optimizer, _continual=False)
    target.load_state_dict(state)
    assert not target.continual()
    assert source.continual()
    assert target.optimizer is target_optimizer
    assert target.max_steps == 3


def test_zero_step_budget():
    optimizer = Mock(spec=_Optimizer)
    scheduler = StopOnPlateau(optimizer, steps=0)
    assert not scheduler.continual()
    scheduler.optimize(input=None)
    optimizer.step.assert_not_called()



def test_zero_patience_stops_only_when_improvement_is_below_threshold():
    optimizer = Mock(spec=_Optimizer)
    scheduler = StopOnPlateau(optimizer, steps=4, patience=0, decreasing=0.25)
    optimizer.last, optimizer.loss = 100.0, 99.5
    scheduler.step(optimizer.loss)
    assert scheduler.continual()
    optimizer.last, optimizer.loss = 99.5, 99.4
    scheduler.step(optimizer.loss)
    assert not scheduler.continual()
