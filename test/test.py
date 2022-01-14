import torch
from torch import nn
import pypose as pp

torch.manual_seed(0)
torch.cuda.manual_seed(0)

x = pp.so3(torch.randn(3,3)*0.1, requires_grad=True)
y = pp.randn_so3(3, sigma=0.1, dtype=torch.float64, requires_grad=True, device="cuda")
a = pp.randn_se3(3, sigma=0.1, requires_grad=True)
b = pp.SE3_type.randn(3, sigma=0.1, requires_grad=True)

assert y.is_leaf and x.is_leaf and a.is_leaf and b.is_leaf

(pp.Log(y.Exp())**2).sin().sum().backward()

print(y)
print(y.shape)
print(y.grad)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class SO3Layer(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.weight = pp.Parameter(pp.randn_so3(n, requires_grad=True), pp.so3_type)

    def forward(self, x):
        return self.weight.Exp() * x

n = 4
epoch = 10
net = SO3Layer(n).cuda()

optimizer = torch.optim.SGD(net.parameters(), lr = 0.1, momentum=0.9)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,15], gamma=0.1)

for i in range(epoch):
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
    inputs = torch.randn(n, 4).to('cuda')
    outputs = net(inputs)
    loss = outputs.abs().sum()
    loss.backward()
    print(loss)

print("Parameter:", count_parameters(net))

SO3I = pp.identity_SO3(1, 3, device="cuda", dtype=torch.float64)
so3I = pp.identity_so3(2, 1, device="cuda", requires_grad=True)

SE3I = pp.identity_SE3(3, device="cuda", dtype=torch.float64)
se3I = pp.identity_se3(2)

print(SO3I)
print(so3I)
print(SE3I)
print(se3I)

