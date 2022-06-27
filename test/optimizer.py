import warnings
import torch, time
import pypose as pp
from torch import nn


class Timer:
    def __init__(self):
        torch.cuda.synchronize()
        self.start_time = time.time()

    def tic(self):
        self.start()

    def show(self, prefix="", output=True):
        torch.cuda.synchronize()
        duration = time.time()-self.start_time
        if output:
            print(prefix+"%fs" % duration)
        return duration

    def toc(self, prefix=""):
        self.end()
        print(prefix+"%fs = %fHz" % (self.duration, 1/self.duration))
        return self.duration

    def start(self):
        torch.cuda.synchronize()
        self.start_time = time.time()

    def end(self, reset=True):
        torch.cuda.synchronize()
        self.duration = time.time()-self.start_time
        if reset:
            self.start_time = time.time()
        return self.duration


import argparse
from torch import nn
import torch.utils.data as Data
from torchvision.datasets import MNIST
from torchvision import transforms as T
parser = argparse.ArgumentParser()
parser.add_argument('--device', default='cuda:0', type=str, help='device')
parser.add_argument('--epoch', default=20, type=int, help='epoch')
parser.add_argument('--batch-size', default=1000, type=int, help='epoch')
parser.add_argument('--damping', default=1e-6, type=float, help='Damping factor')
parser.add_argument('--gamma', default=2, type=float, help='Gamma')
args = parser.parse_args()


class PoseInv(nn.Module):
    def __init__(self, *dim):
        super().__init__()
        self.pose = pp.Parameter(pp.randn_rxso3(*dim))

    def forward(self, inputs):
        return (self.pose.Exp() @ inputs).Log()


posnet = PoseInv(2, 2)
inputs = pp.randn_RxSO3(2, 2)
target = pp.identity_rxso3(2, 2)
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


posnet = PoseInv(2, 2)
optimizer = pp.optim.LM(posnet, damping=args.damping)
timer = Timer()

for idx in range(10):
    loss = optimizer.step(inputs, target)
    print('Pose loss %.7f @ %dit, Timing: %.3fs'%(loss, idx, timer.end()))
    if loss < 1e-5:
        print('Early Stoping!')
        print('Optimization Early Done with loss:', loss.sum().item())
        break


class PoseInv(nn.Module):
    def __init__(self, *dim):
        super().__init__()
        self.pose = pp.Parameter(pp.randn_RxSO3(*dim))

    def forward(self, inputs):
        return (self.pose @ inputs).Log()

posnet = PoseInv(2, 2)
optimizer = pp.optim.LM(posnet, damping=1e-6)

for idx in range(10):
    loss = optimizer.step(inputs, target)
    print('Pose loss %.7f @ %dit, Timing: %.3fs'%(loss, idx, timer.end()))
    if loss < 1e-5:
        print('Early Stoping!')
        print('Optimization Early Done with loss:', loss.sum().item())
        break
