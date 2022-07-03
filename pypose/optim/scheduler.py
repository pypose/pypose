from torch.optim.lr_scheduler import _LRScheduler


class UpDownDampingScheduler(_LRScheduler):
    '''
    UpDownDampingScheduler for LevenbergMarquardt Optimizer
    '''
    def __init__(self, optimizer, gamma, verbose=False):
        assert gamma > 1, 'Invalid Gamma.'
        self.gamma, self.loss = gamma, float('Inf')
        super().__init__(optimizer, verbose=verbose, last_epoch=-1)

    def get_lr(self):
        factor = 1/self.gamma if self.optimizer.loss < self.loss else self.gamma
        self.loss = self.optimizer.loss
        self.optimizer.loss = 0
        return [group['lr']*factor for group in self.optimizer.param_groups]
