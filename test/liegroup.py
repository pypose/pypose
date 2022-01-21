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
        return self.weight * x

n = 4
epoch = 10
net = SO3Layer(n).cuda()

optimizer = torch.optim.SGD(net.parameters(), lr = 0.1, momentum=0.9)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,15], gamma=0.1)

for i in range(epoch):
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
    inputs = torch.randn(n, 4, device="cuda").sum()
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

idl1 = pp.identity_like(x)
idl2 = pp.identity_like(a, requires_grad=True)

rdl1 = pp.randn_like(x)
rdl2 = pp.randn_like(a, sigma=0.1, requires_grad=True, device='cuda')

print(rdl1, '\n', rdl2)

b.Inv()
print(b,'\n', b.Inv())
print(x,'\n', x.Inv())

print(pp.Inv(b))

a = pp.randn_SE3(1,5)
b = pp.randn_SE3(5,1)
c = pp.randn_SO3(1,5)

e = a * b
f = a * c

g = pp.randn_so3(1,5)
g * 0.1

pp.Mul(a, b)

points = torch.randn(2,5,3)

pt = a * points

print(pt.shape)

a = pp.randn_so3(3)
I = pp.identity_SO3(3)
t = I.Retr(a)

I = pp.identity_SE3(3)
t = I.Retr(a)

r = pp.Retr(I, a)
p = I.Act(a)

print(r, p)

X = pp.randn_SE3(8, requires_grad=True, dtype=torch.double)
a = pp.randn_se3(8, dtype=torch.double)
b = X.Adj(a)
assert (b.Exp() * X - X * a.Exp()).abs().mean() < 1e-7
J = X.Jinv(a)
print(J)

X = pp.randn_SO3(6, requires_grad=True, dtype=torch.double)
a = pp.randn_so3(6, dtype=torch.double)
b = X.AdjT(a)
assert (X * b.Exp() - a.Exp() * X).abs().mean() < 1e-7

J = pp.Jinv(X, a)
print(J)


S = pp.randn_Sim3(4)
S.Log().Exp()

s = pp.randn_sim3(4)
s.Exp().Log()

X = pp.randn_Sim3(8, requires_grad=True, dtype=torch.double)
a = pp.randn_sim3(8, dtype=torch.double)
b = X.Adj(a)
assert (b.Exp() * X - X * a.Exp()).abs().mean() < 1e-7

X = pp.randn_Sim3(6, requires_grad=True, dtype=torch.double)
a = pp.randn_sim3(6, dtype=torch.double)
b = X.AdjT(a)
print((X * b.Exp() - a.Exp() * X).abs().mean())

S = pp.randn_RxSO3(6)
s = pp.randn_rxso3(6)
s.Exp() * S

X = pp.randn_SE3(8, requires_grad=True, dtype=torch.double)
print(X.matrix())
print(X.translation())
assert hasattr(X.view(2,4,7), 'gtype')
X.gview(2,4)
print(X)


X = pp.randn_SE3(2, requires_grad=True, dtype=torch.double)
Y = pp.randn_SE3(2, requires_grad=True, dtype=torch.double)
from torch.overrides import get_overridable_functions
func_dict = get_overridable_functions()

Z = torch.cat([X,Y], dim=0)
assert isinstance(Z, pp.LieGroup)
assert isinstance(torch.stack([X,Y], dim=0), pp.LieGroup)

print(torch.stack([X,Y], dim=0))
Z1, Z2 = Z.split([2,2], dim=0)
assert isinstance(Z1, pp.LieGroup)
assert isinstance(Z2, pp.LieGroup)
assert not isinstance(torch.randn(4), pp.LieGroup)

a, b = Z.tensor_split(2)
assert isinstance(a, pp.LieGroup) and isinstance(b, pp.LieGroup)