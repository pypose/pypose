import torch


class _Planner(object):

    def __init__(self, max_steps, verbose=False):
        self.max_steps, self.verbose = max_steps, verbose
        self.reset()

    def continual(self):
        return self._continual

    def reset(self):
        self.last = torch.tensor(float('inf'))
        self.steps, self._continual = 0, True


class ReduceToBason(_Planner):
    r'''
    A scheduler to stop a loop when no relative loss 'decreasing' is seen for a 'patience'
    number of steps.

    Args:
        steps (int): Maximum number of interations optimizer will step.
        patience (int): Number of steps with no loss 'decreasing' is seen. For example, if
            ``patience = 2``, then it ignores the first 2 steps with no improvement, and
            stop the loop after the 3rd step if the loss has no decerasing. Default: 5.
        decreasing (float): relative loss decreasing used to count the number of patience
            steps. Default: 1e-3.
        verbose (bool): if ``True``, prints a message to stdout for each step.
            Default: ``False``.

    Warning:
        Remember to call `scheduler.reset()` if you want to re-use a scheduler.

    Example:
        >>> from pypose.module import ReduceToPlateau
        >>> x = torch.tensor(0.9)
        >>> scheduler = ReduceToPlateau(steps=5, patience=2, decreasing=0.1, verbose=True)
        >>> while scheduler.continual():
        ...     x = x ** 2
        ...     scheduler.step(x)
        ReduceToPlateau step 0 loss 8.099999e-01.
        ReduceToPlateau step 1 loss 6.560999e-01.
        ReduceToPlateau step 2 loss 4.304671e-01.
        ReduceToPlateau step 3 loss 1.853019e-01.
        ReduceToPlateau step 4 loss 3.433681e-02.
        ReduceToPlateau: Maximum steps reached, Quiting..
    '''
    def __init__(self, steps, patience=5, decreasing=1e-3, verbose=False):
        super().__init__(steps, verbose)
        self.decreasing = decreasing
        self.patience, self.patience_count = patience, 0

    def step(self, loss):
        r'''
        Performs a scheduler step.

        Args:
            loss (``float`` or ``torch.Tensor``): the model loss after one loop.
                Can be a batched tensor. If batched tensor, all losses in the batch has to
                satisfy the condition to stop a loop.
        '''
        if self.verbose:
            print('ReduceToPlateau step', self.steps, 'loss', loss)

        if not torch.is_tensor(loss):
            loss = torch.tensor(loss)

        self.steps = self.steps + 1

        if self.steps >= self.max_steps:
            self._continual = False
            if self.verbose:
                print("ReduceToPlateau: Maximum steps reached, Quiting..")

        if torch.all((self.last - loss)/loss < self.decreasing):
            self.patience_count = self.patience_count + 1
        else:
            self.patience_count = 0

        self.last = loss

        if self.patience_count >= self.patience:
            self._continual = False
            if self.verbose:
                print("ReduceToPlateau: Maximum patience steps reached, Quiting..")
