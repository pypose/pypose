import time
import torch
import warnings
import pypose as pp
from torch import nn
import pypose.optim.solver as ppos
import pypose.optim.kernel as ppok
import pypose.optim.corrector as ppoc
import pypose.optim.strategy as ppst
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Timer:
    def __init__(self):
        self.synchronize()
        self.start_time = time.time()
    
    def synchronize(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()  

    def tic(self):
        self.start()

    def show(self, prefix="", output=True):
        self.synchronize()
        duration = time.time()-self.start_time
        if output:
            print(prefix+"%fs" % duration)
        return duration

    def toc(self, prefix=""):
        self.end()
        print(prefix+"%fs = %fHz" % (self.duration, 1/self.duration))
        return self.duration

    def start(self):
        self.synchronize()
        self.start_time = time.time()

    def end(self, reset=True):
        self.synchronize()
        self.duration = time.time()-self.start_time
        if reset:
            self.start_time = time.time()
        return self.duration


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

        while scheduler.continual:
            loss = optimizer.step(inputs)
            scheduler.step(loss)


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
