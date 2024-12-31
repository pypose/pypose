import time
import torch
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
        assert idx == 99, "Optimization shouldn't stop early."

        posnet = PoseInv(2, 2).to(device)
        optimizer = pp.optim.LM(posnet)

        for idx in range(10):
            loss = optimizer.step(inputs, target)
            print('Pose loss %.7f @ %dit, Timing: %.3fs'%(loss, idx, timer.end()))
            if loss < 1e-5:
                print('Early Stoping!')
                print('Optimization Early Done with loss:', loss.sum().item())
                break
        print('Done')
        assert idx < 9, "Optimization requires too many steps."


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
        info = torch.eye(4, device=device)
        optimizer = pp.optim.LM(posnet, weight=info)

        for idx in range(10):
            loss = optimizer.step(inputs, target)
            print('Pose loss %.7f @ %dit, Timing: %.3fs'%(loss, idx, timer.end()))
            if loss < 1e-5:
                print('Early Stoping!')
                print('Optimization Early Done with loss:', loss.sum().item())
                break

        assert idx < 9, "Optimization requires too many steps."


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
        optimizer = pp.optim.LM(posnet, solver=solver, kernel=kernel, corrector=corrector)

        for idx in range(10):
            loss = optimizer.step(inputs, target)
            print('Pose loss %.7f @ %dit, Timing: %.3fs'%(loss, idx, timer.end()))
            if loss < 1e-5:
                print('Early Stoping!')
                print('Optimization Early Done with loss:', loss.sum().item())
                break

        assert idx < 9, "Optimization requires too many steps."

    def test_optim_strategy_constant(self):

        class PoseInv(nn.Module):
            def __init__(self, *dim):
                super().__init__()
                self.pose = pp.Parameter(pp.randn_SE3(*dim))

            def forward(self, inputs):
                return (self.pose @ inputs).Log().tensor()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputs = pp.randn_SE3(2, 2).to(device)
        invnet = PoseInv(2, 2).to(device)
        strategy = pp.optim.strategy.Constant(damping=1e-6)
        optimizer = pp.optim.LM(invnet, strategy=strategy)

        for idx in range(10):
            loss = optimizer.step(inputs)
            print('Pose loss %.7f @ %dit'%(loss, idx))
            if loss < 1e-5:
                print('Early Stoping!')
                print('Optimization Early Done with loss:', loss.item())
                break

        assert idx < 9, "Optimization requires too many steps."

    def test_optim_strategy_adaptive(self):

        class PoseInv(nn.Module):
            def __init__(self, *dim):
                super().__init__()
                self.pose = pp.Parameter(pp.randn_SE3(*dim))

            def forward(self, inputs):
                return (self.pose @ inputs).Log().tensor()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputs = pp.randn_SE3(2, 2).to(device)
        invnet = PoseInv(2, 2).to(device)
        strategy = pp.optim.strategy.Adaptive(damping=1e-6)
        optimizer = pp.optim.LM(invnet, strategy=strategy)

        for idx in range(10):
            loss = optimizer.step(inputs)
            print('Pose loss %.7f @ %dit'%(loss, idx))
            if loss < 1e-5:
                print('Early Stoping!')
                print('Optimization Early Done with loss:', loss.item())
                break

        assert idx < 9, "Optimization requires too many steps."

    def test_optim_trustregion(self):
        class PoseInv(nn.Module):
            def __init__(self, *dim):
                super().__init__()
                self.pose = pp.Parameter(pp.randn_SE3(*dim))

            def forward(self, inputs):
                return (self.pose @ inputs).Log().tensor()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputs = pp.randn_SE3(2, 2).to(device)
        invnet = PoseInv(2, 2).to(device)
        strategy = pp.optim.strategy.TrustRegion(radius=1e6)
        optimizer = pp.optim.LM(invnet, strategy=strategy)

        for idx in range(10):
            loss = optimizer.step(inputs)
            print('Pose loss %.7f @ %dit'%(loss, idx))
            if loss < 1e-5:
                print('Early Stoping!')
                print('Optimization Early Done with loss:', loss.item())
                break

        assert idx < 9, "Optimization requires too many steps."

    def test_optim_multiparameter(self):
        class PoseInv(nn.Module):
            def __init__(self, *dim):
                super().__init__()
                self.pose1 = pp.Parameter(pp.randn_SE3(*dim))
                self.pose2 = pp.Parameter(pp.randn_SE3(*dim))

            def forward(self, inputs):
                return ((self.pose1 @ inputs).Log().tensor() + (self.pose2 @ inputs).Log().tensor())

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputs = pp.randn_SE3(2, 2).to(device)
        invnet = PoseInv(2, 2).to(device)
        strategy = pp.optim.strategy.TrustRegion(radius=1e6)
        optimizer = pp.optim.LM(invnet, strategy=strategy)

        for idx in range(10):
            loss = optimizer.step(inputs)
            print('Pose loss %.7f @ %dit'%(loss, idx))
            if loss < 1e-5:
                print('Early Stoping!')
                print('Optimization Early Done with loss:', loss.item())
                break

        assert idx < 9, "Optimization requires too many steps."

    def test_optim_anybatch(self):

        class PoseInv(nn.Module):
            def __init__(self, *dim):
                super().__init__()
                self.pose1 = pp.Parameter(pp.randn_SE3(*dim))
                self.pose2 = pp.Parameter(pp.randn_SE3(*dim))

            def forward(self, inputs):
                return ((self.pose1 @ inputs).Log().tensor() + (self.pose2 @ inputs).Log().tensor())

        B1, B2, M, N = 2, 3, 2, 2
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputs = pp.randn_SE3(B2, B1, M, N, sigma=0.0001).to(device)
        invnet = PoseInv(M, N).to(device)
        strategy = pp.optim.strategy.TrustRegion(radius=1e6)
        optimizer = pp.optim.LM(invnet, strategy=strategy)

        for idx in range(10):
            loss = optimizer.step(inputs)
            print('Pose loss %.7f @ %dit'%(loss, idx))
            if loss < 1e-5:
                print('Early Stoping!')
                print('Optimization Early Done with loss:', loss.item())
                break

        assert idx < 9, "Optimization requires too many steps."

    def test_optim_multi_input(self):

        class PoseInv(nn.Module):
            def __init__(self, *dim):
                super().__init__()
                self.pose = pp.Parameter(pp.randn_SE3(*dim))

            def forward(self, poses):
                error1 = (self.pose @ poses).Log().tensor()
                error2 = self.pose.Log().tensor().sum(-1, keepdim=True)
                return error1, error2

        B1, B2, M, N = 2, 3, 2, 2
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputs = pp.randn_SE3(B2, B1, M, N, sigma=0.0001).to(device)
        invnet = PoseInv(M, N).to(device)
        strategy = pp.optim.strategy.TrustRegion(radius=1e6)
        kernel = [ppok.Huber().to(device), ppok.Scale().to(device)]
        weight = [torch.eye(6, device=device), torch.ones(1, device=device)]
        optimizer = pp.optim.LM(invnet, strategy=strategy, kernel=kernel)
        # optimizer = pp.optim.GN(invnet, kernel=kernel)

        for idx in range(10):
            loss = optimizer.step(input={'poses':inputs}, weight=weight)
            print('Pose loss %.7f @ %dit'%(loss, idx))
            if loss < 1e-5:
                print('Early Stoping!')
                print('Optimization Early Done with loss:', loss.item())
                break

        assert idx < 9, "Optimization requires too many steps."


    def test_batch_weight(self):

        class PoseInv(nn.Module):
            def __init__(self, *dim):
                super().__init__()
                self.pose = pp.Parameter(pp.randn_SE3(*dim))

            def forward(self, poses):
                error1 = (self.pose @ poses).Log().tensor()
                error2 = self.pose.Log().tensor().sum(-1, keepdim=True)
                return error1, error2

        B1, B2, M, N = 2, 3, 2, 2
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputs = pp.randn_SE3(B2, B1, M, N, sigma=0.0001).to(device)
        invnet = PoseInv(M, N).to(device)
        strategy = pp.optim.strategy.TrustRegion(radius=1e6)
        kernel = [ppok.Huber().to(device), ppok.Scale().to(device)]
        weight = [torch.eye(6, device=device).unsqueeze(0).repeat(N, 1, 1), \
                  torch.ones(1, device=device).unsqueeze(0)]
        optimizer = pp.optim.LM(invnet, strategy=strategy, kernel=kernel)

        for idx in range(10):
            loss = optimizer.step(input={'poses':inputs}, weight=weight)
            print('Pose loss %.7f @ %dit'%(loss, idx))
            if loss < 1e-5:
                print('Early Stoping!')
                print('Optimization Early Done with loss:', loss.item())
                break

        assert idx < 9, "Optimization requires too many steps."


if __name__ == '__main__':
    test = TestOptim()
    test.test_optim_liealgebra()
    test.test_optim_liegroup()
    test.test_optim_with_kernel()
    test.test_optim_strategy_constant()
    test.test_optim_strategy_adaptive()
    test.test_optim_trustregion()
    test.test_optim_multiparameter()
    test.test_optim_anybatch()
    test.test_optim_multi_input()
    test.test_batch_weight()
