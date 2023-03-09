import torch
from .optimizer import _Optimizer


class _Scheduler(object):
    class Continual:
        """
        From PyPose v0.3.6, we change scheduler.continual to scheduler.continual().
        This is a temporary workaround for triggering an error when users call continual
        attribute of the scheduler. This wrapper will be removed in a future release.
        """
        def __init__(self, optimizer):
            self.optimizer = optimizer

        def __call__(self, *args, **kwargs):
            '''
            Determining whether to stop an optimizer should be provided here.
            This function is only for temporarailiy replacing Scheduler.continual().
            '''
            return self.optimizer.iscontinual(*args, **kwargs)

        def __bool__(self):
            raise RuntimeError('Calling scheduler.continual is deprecated, '
                               'please call scheduler.continual() instead. '
                               'This error msg will be removed in a future release.')

    def __init__(self, optimizer, max_steps, verbose=False):

        # Attach optimizer
        if not isinstance(optimizer, _Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))

        self.optimizer, self.verbose = optimizer, verbose
        self.max_steps, self.steps = max_steps, 0
        self.continual = self.Continual(self)
        self._continual = True

    def iscontinual(self):
        '''
        This is a temporary function.
        We will change to continual() in a future release.
        '''
        return self._continual

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
    r'''
    A scheduler to stop an optimizer when no relative loss 'decreasing' is seen for a 'patience'
    number of steps.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        steps (int): Maximum number of interations optimizer will step.
        patience (int): Number of steps with no loss 'decreasing' is seen. For example, if
            ``patience = 2``, then it ignores the first 2 steps with no improvement, and stop
            the optimizer after the 3rd step if the loss has no decerasing. Default: 5.
        decreasing (float): relative loss decreasing used to count the number of patience steps.
            Default: 1e-3.
        verbose (bool): if ``True``, prints a message to stdout for each step. Default: ``False``.

    Note:
        The users have two options to use a scheduler.
        The first one is to call the :meth:`step` method for multiple times, which is easy to be
        extended for customized operation in each iteration.
        The second one is to call the :meth:`optimize` method, which interally calls :meth:`step`
        multiple times to perform full optimization.
        See examples below.
    '''
    def __init__(self, optimizer, steps, patience=5, decreasing=1e-3, verbose=False):
        super().__init__(optimizer, steps, verbose)
        self.decreasing = decreasing
        self.patience, self.patience_count = patience, 0

    def step(self, loss):
        r'''
        Performs a scheduler step.

        Args:
            loss (float): the model loss after one optimizer step.

        Example:

            >>> class PoseInv(nn.Module):
            ...
            ...     def __init__(self, *dim):
            ...         super().__init__()
            ...         self.pose = pp.Parameter(pp.randn_SE3(*dim))
            ...
            ...     def forward(self, input):
            ...         return (self.pose @ input).Log().tensor()
            ...
            >>> device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            >>> input = pp.randn_SE3(2, 2).to(device)
            >>> invnet = PoseInv(2, 2).to(device)
            >>> strategy = pp.optim.strategy.Constant(damping=1e-4)
            >>> optimizer = pp.optim.LM(invnet, strategy=strategy)
            >>> scheduler = pp.optim.scheduler.StopOnPlateau(optimizer, steps=10, \
            >>>                     patience=3, decreasing=1e-3, verbose=True)
            ...
            >>> while scheduler.continual():
            ...     loss = optimizer.step(input)
            ...     scheduler.step(loss)
            StopOnPlateau on step 0 Loss 9.337769e+01 --> Loss 3.502787e-05 (reduction/loss: 1.0000e+00).
            StopOnPlateau on step 1 Loss 3.502787e-05 --> Loss 4.527339e-13 (reduction/loss: 1.0000e+00).
            StopOnPlateau on step 2 Loss 4.527339e-13 --> Loss 7.112640e-14 (reduction/loss: 8.4290e-01).
            StopOnPlateau on step 3 Loss 7.112640e-14 --> Loss 3.693307e-14 (reduction/loss: 4.8074e-01).
            StopOnPlateau: Maximum patience steps reached, Quiting..
        '''
        assert self.optimizer.loss is not None, \
            'scheduler.step() should be called after optimizer.step()'

        if self.verbose:
            print('StopOnPlateau on step {} Loss {:.6e} --> Loss {:.6e} '\
                  '(reduction/loss: {:.4e}).'.format(self.steps, self.optimizer.last,
                  self.optimizer.loss, (self.optimizer.last - self.optimizer.loss) /\
                  (self.optimizer.last + 1e-31)))

        self.steps = self.steps + 1

        if self.steps >= self.max_steps:
            self._continual = False
            if self.verbose:
                print("StopOnPlateau: Maximum steps reached, Quiting..")

        if (self.optimizer.last - self.optimizer.loss) < self.decreasing:
            self.patience_count = self.patience_count + 1
        else:
            self.patience_count = 0

        if self.patience_count >= self.patience:
            self._continual = False
            if self.verbose:
                print("StopOnPlateau: Maximum patience steps reached, Quiting..")

        if hasattr(self.optimizer, 'reject_count'):
            if self.optimizer.reject_count > 0:
                self._continual = False
                if self.verbose:
                    print("StopOnPlateau: Maximum rejected steps reached, Quiting..")

    @torch.no_grad()
    def optimize(self, input, target=None, weight=None):
        r'''
        Perform full optimization steps.

        Args:
            input (Tensor/LieTensor or tuple of Tensors/LieTensors): the input to the model.
            target (Tensor/LieTensor): the model target to optimize.
                If not given, the squared model output is minimized. Defaults: ``None``.
            weight (Tensor, optional): a square positive definite matrix defining the weight of
                model residual. Default: ``None``.

        The above arguments are sent to optimizers. More details go to
        :obj:`pypose.optim.LevenbergMarquardt` or :obj:`pypose.optim.GaussNewton`.

        Example:
            >>> class PoseInv(nn.Module):
            ...
            ...     def __init__(self, *dim):
            ...         super().__init__()
            ...         self.pose = pp.Parameter(pp.randn_SE3(*dim))
            ...
            ...     def forward(self, input):
            ...         return (self.pose @ input).Log().tensor()
            ...
            >>> device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            >>> input = pp.randn_SE3(2, 2).to(device)
            >>> invnet = PoseInv(2, 2).to(device)
            >>> strategy = pp.optim.strategy.Constant(damping=1e-4)
            >>> optimizer = pp.optim.LM(invnet, strategy=strategy)
            >>> scheduler = pp.optim.scheduler.StopOnPlateau(optimizer, steps=10, \
            >>>                     patience=3, decreasing=1e-3, verbose=True)
            ...
            >>> scheduler.optimize(input=input)
            StopOnPlateau on step 0 Loss 5.199298e+01 --> Loss 8.425808e-06 (reduction/loss: 1.0000e+00).
            StopOnPlateau on step 1 Loss 8.425808e-06 --> Loss 3.456247e-13 (reduction/loss: 1.0000e+00).
            StopOnPlateau on step 2 Loss 3.456247e-13 --> Loss 1.525355e-13 (reduction/loss: 5.5867e-01).
            StopOnPlateau on step 3 Loss 1.525355e-13 --> Loss 6.769275e-14 (reduction/loss: 5.5622e-01).
            StopOnPlateau: Maximum patience steps reached, Quiting..
        '''
        while self.continual():
            loss = self.optimizer.step(input, target, weight)
            self.step(loss)
