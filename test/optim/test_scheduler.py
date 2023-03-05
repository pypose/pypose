import torch
import pypose as pp
from torch import nn


class TestScheduler:

    def test_optim_scheduler(self):

        class PoseInv(nn.Module):

            def __init__(self, *dim):
                super().__init__()
                self.pose = pp.Parameter(pp.randn_SE3(*dim))

            def forward(self, inputs):
                return (self.pose @ inputs).Log().tensor()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputs = pp.randn_SE3(2, 2).to(device)
        invnet = PoseInv(2, 2).to(device)
        strategy = pp.optim.strategy.Constant(damping=1e2)
        optimizer = pp.optim.LM(invnet, strategy=strategy)
        scheduler = pp.optim.scheduler.StopOnPlateau(optimizer, steps=100, \
                            patience=5, decreasing=1, verbose=True)

        while scheduler.continual():
            loss = optimizer.step(inputs)
            scheduler.step(loss)

        # the following block should trigger a runtime error
        runtime_error = False
        try:
            while scheduler.continual:
                loss = optimizer.step(inputs)
                scheduler.step(loss)
        except RuntimeError:
            runtime_error = True
        assert runtime_error



    def test_optim_scheduler_optimize(self):

        class PoseInv(nn.Module):

            def __init__(self, *dim):
                super().__init__()
                self.pose = pp.Parameter(pp.randn_SE3(*dim))

            def forward(self, inputs):
                return (self.pose @ inputs).Log().tensor()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputs = pp.randn_SE3(2, 2).to(device)
        invnet = PoseInv(2, 2).to(device)
        strategy = pp.optim.strategy.Constant(damping=1e-7)
        optimizer = pp.optim.LM(invnet, strategy=strategy)
        scheduler = pp.optim.scheduler.StopOnPlateau(optimizer, steps=30, \
            patience=5, decreasing=0, verbose=True)

        scheduler.optimize(input=inputs)

    def test_optim_scheduler_optimize_gn(self):

        class PoseInv(nn.Module):

            def __init__(self, *dim):
                super().__init__()
                self.pose = pp.Parameter(pp.randn_SE3(*dim))

            def forward(self, inputs):
                return (self.pose @ inputs).Log().tensor()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputs = pp.randn_SE3(2, 2).to(device)
        invnet = PoseInv(2, 2).to(device)
        optimizer = pp.optim.GN(invnet)
        scheduler = pp.optim.scheduler.StopOnPlateau(optimizer, steps=100, \
            patience=5, decreasing=1e-2, verbose=True)

        scheduler.optimize(input=inputs)


if __name__ == '__main__':
    test = TestScheduler()
    test.test_optim_scheduler()
    test.test_optim_scheduler_optimize()
    test.test_optim_scheduler_optimize_gn()
