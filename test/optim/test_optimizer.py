import time
import torch
import warnings
import pypose as pp
from torch import nn
import pypose.optim.solver as ppos
import pypose.optim.kernel as ppok
import pypose.optim.corrector as ppoc
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


class TestOptim:
    def test_optim_liealgebra(self):

        class PoseInv(nn.Module):
            def __init__(self, *dim):
                super().__init__()
                self.pose = pp.Parameter(pp.randn_rxso3(*dim))

            def forward(self, inputs):
                return (self.pose.Exp() @ inputs).Log()

        posnet = PoseInv(2, 2).to(device)
        inputs = pp.randn_RxSO3(2, 2).to(device)
        target = pp.identity_rxso3(2, 2).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(posnet.parameters(), lr=1e-1)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [50, 70], gamma=0.1)
        timer = Timer()

        for idx in range(100):
            optimizer.zero_grad()
            output = posnet(inputs)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            scheduler.step()
            print('Pose loss %.7f @ %dit, Timing: %.3fs'%(loss.item(), idx, timer.end()))
            if loss.sum() < 1e-5:
                print('Early Stoping!')
                print('Optimization Early Done with loss:', loss.item())
                break
        print('Done', timer.toc())
        assert idx == 99

        posnet = PoseInv(2, 2).to(device)
        optimizer = pp.optim.LM(posnet, damping=1e-6)

        for idx in range(10):
            loss = optimizer.step(inputs, target)
            print('Pose loss %.7f @ %dit, Timing: %.3fs'%(loss, idx, timer.end()))
            if loss < 1e-5:
                print('Early Stoping!')
                print('Optimization Early Done with loss:', loss.sum().item())
                break
        print('Done')
        assert idx < 10


    def test_optim_liegroup(self):

        class PoseInv(nn.Module):
            def __init__(self, *dim):
                super().__init__()
                self.pose = pp.Parameter(pp.randn_RxSO3(*dim))

            def forward(self, inputs):
                return (self.pose @ inputs).Log()

        timer = Timer()
        inputs = pp.randn_RxSO3(2, 2).to(device)
        target = pp.identity_rxso3(2, 2).to(device)
        posnet = PoseInv(2, 2).to(device)
        optimizer = pp.optim.LM(posnet, damping=1e-6)

        for idx in range(10):
            loss = optimizer.step(inputs, target)
            print('Pose loss %.7f @ %dit, Timing: %.3fs'%(loss, idx, timer.end()))
            if loss < 1e-5:
                print('Early Stoping!')
                print('Optimization Early Done with loss:', loss.sum().item())
                break
        
        assert idx < 10


    def test_optim_with_kernel(self):

        class PoseInv(nn.Module):
            def __init__(self, *dim):
                super().__init__()
                self.pose = pp.Parameter(pp.randn_SE3(*dim))

            def forward(self, inputs):
                return (self.pose @ inputs).Log()

        timer = Timer()
        target = pp.identity_se3(2, 2).to(device)
        inputs = pp.randn_SE3(2, 2).to(device)
        posnet = PoseInv(2, 2).to(device)
        solver = ppos.Cholesky()
        kernel = ppok.Cauchy()
        corrector = ppoc.FastTriggs(kernel)
        optimizer = pp.optim.LM(posnet, damping=1e-6, solver=solver, kernel=kernel, corrector=corrector)

        for idx in range(10):
            loss = optimizer.step(inputs, target)
            print('Pose loss %.7f @ %dit, Timing: %.3fs'%(loss, idx, timer.end()))
            if loss < 1e-5:
                print('Early Stoping!')
                print('Optimization Early Done with loss:', loss.sum().item())
                break

        assert idx < 10


if __name__ == '__main__':
    test = TestOptim()
    test.test_optim_liealgebra()
    test.test_optim_liegroup()
    test.test_optim_with_kernel()
