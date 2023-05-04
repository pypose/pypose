import torch


class _Stepper(object):

    def __init__(self, max_steps, verbose=False):
        self.max_steps, self.verbose = max_steps, verbose
        self.reset()

    def continual(self):
        return self._continual

    def reset(self):
        self.last = torch.tensor(float('inf'))
        self.steps, self._continual = 0, True


class ReduceToBason(_Stepper):
    r'''
    A stepper to stop a loop when no relative loss 'decreasing' is seen for a 'patience'
    number of steps.

    Args:
        steps (``int``): Maximum number of iterations a loop will step.
        patience (``int``, optional): Number of steps with no loss 'decreasing' is seen.
            For example, if ``patience = 2``, then it ignores the first 2 steps with no
            improvement, and stop the loop after the 3rd step if the loss has no
            decerasing. Default: ``5``.
        decreasing (``float``, optional): relative loss decreasing used to count the
            number of patience steps. Default: ``1e-3``.
        tol (``float``, optional): the minimum loss tolerance to stop a loop.
            Default: ``1e-5``.
        verbose (``bool``, optional): if ``True``, prints a message to stdout for each
            step. Default: ``False``.

    Warning:
        Remember to call `stepper.reset()` if you want to re-use a stepper.

    Example:
        >>> from pypose.utils import ReduceToBason
        >>> x = 0.9
        >>> stepper = ReduceToBason(steps=5, patience=2, decreasing=0.1, verbose=True)
        >>> while stepper.continual():
        ...     x = x ** 2
        ...     stepper.step(x)
        ReduceToBason step 0 loss 8.099999e-01.
        ReduceToBason step 1 loss 6.560999e-01.
        ReduceToBason step 2 loss 4.304671e-01.
        ReduceToBason step 3 loss 1.853019e-01.
        ReduceToBason step 4 loss 3.433681e-02.
        ReduceToBason: Maximum steps reached, Quiting..
    '''
    def __init__(self, steps, patience=5, decreasing=1e-3, tol=1e-5, verbose=False):
        super().__init__(steps, verbose)
        self.decreasing, self.tol = decreasing, tol
        self.patience, self.patience_count = patience, 0

    def step(self, loss):
        r'''
        Performs a stepper step.

        Args:
            loss (``float`` or ``torch.Tensor``): the model loss after one loop.
                Can be a batched tensor. If batched tensor, all losses in the batch has to
                satisfy the condition to stop a loop.
        '''
        if self.verbose:
            print('ReduceToBason step', self.steps, 'loss', loss)

        if not torch.is_tensor(loss):
            loss = torch.tensor(loss)

        self.steps = self.steps + 1

        if torch.all(loss < self.tol):
            self._continual = False
            if self.verbose:
                print("ReduceToBason: Loss tol reached, Quiting..")

        if self.steps >= self.max_steps:
            self._continual = False
            if self.verbose:
                print("ReduceToBason: Maximum steps reached, Quiting..")

        if torch.all((self.last - loss)/loss < self.decreasing):
            self.patience_count = self.patience_count + 1
        else:
            self.patience_count = 0

        self.last = loss

        if self.patience_count >= self.patience:
            self._continual = False
            if self.verbose:
                print("ReduceToBason: Maximum patience steps reached, Quiting..")
