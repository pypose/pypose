import torch
from .optimizer import _Optimizer


class _Scheduler(object):

    def __init__(self, optimizer, max_steps, verbose=False):

        # Attach optimizer
        if not isinstance(optimizer, _Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))

        self.optimizer, self.verbose = optimizer, verbose
        self.max_steps, self.steps = max_steps, 0
        self.continual = True

    @property
    def continual(self):
        return self._continual

    @continual.setter
    def continual(self, value):
        assert isinstance(value, bool)
        self._continual = value

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)


class StopOnPlateau(_Scheduler):

    def __init__(self, optimizer, steps, patience=5, decreasing=1e-3, verbose=False):
        super().__init__(optimizer, steps, verbose)
        self.decreasing = decreasing
        self.patience, self.patience_count = patience, 0

    def step(self, loss):

        assert self.optimizer.loss is not None, \
            'scheduler.step() should be called after optimizer.step()'

        if self.verbose:
            print('StopOnPlateau on step {} Loss {:.6e} --> Loss {:.6e} '\
                  '(reduction/loss: {:.4e}).'.format(self.steps, self.optimizer.last,
                  self.optimizer.loss,
                  (self.optimizer.last - self.optimizer.loss) / self.optimizer.last))

        self.steps = self.steps + 1

        if self.steps >= self.max_steps:
            self.continual = False
            if self.verbose:
                print("StopOnPlateau: Maximum steps reached, Quiting..")

        if (self.optimizer.last - self.optimizer.loss) < self.decreasing:
            self.patience_count = self.patience_count + 1
        else:
            self.patience_count = 0

        if self.patience_count >= self.patience:
            self.continual = False
            if self.verbose:
                print("StopOnPlateau: Maximum patience steps reached, Quiting..")

        if hasattr(self.optimizer, 'reject_count'):
            if self.optimizer.reject_count > 0:
                self.continual = False
                if self.verbose:
                    print("StopOnPlateau: Maximum rejected steps reached, Quiting..")

    @torch.no_grad()
    def optimize(self, input, target=None, weight=None, scheduler=None):
        while self.continual:
            loss = self.optimizer.step(input, target, weight)
            self.step(loss)
