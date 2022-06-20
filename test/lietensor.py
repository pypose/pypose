import torch
from torch import nn
import pypose as pp

torch.manual_seed(0)
torch.cuda.manual_seed(0)

x = pp.so3(torch.randn(3,3)*0.1)
y = pp.randn_so3(3, sigma=0.1, dtype=torch.float64, requires_grad=True, device="cuda")
a = pp.randn_se3(3, sigma=0.1, requires_grad=True)
b = pp.SE3_type.randn(3, sigma=0.1, requires_grad=True)

assert y.is_leaf and x.is_leaf and a.is_leaf and b.is_leaf

(pp.Log(y.Exp())**2).sin().sum().backward()

print(y)
print(y.lshape)
print(y.grad)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class SO3Layer(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.weight = pp.Parameter(pp.randn_so3(n))

    def forward(self, x):
        return self.weight.Exp() * x

n = 4
epoch = 3
net = SO3Layer(n).cuda()
print(net.weight)
net.weight.data.add_(0.1)
print(net.weight)
optimizer = torch.optim.SGD(net.parameters(), lr = 0.1, momentum=0.9)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,15], gamma=0.1)
print(net.weight)
for i in range(epoch):
    optimizer.zero_grad()
    inputs = pp.randn_SO3(n, device="cuda")
    outputs = net(inputs)
    loss = outputs.abs().sum()
    loss.backward()
    optimizer.step()
    scheduler.step()
    print(loss)
print(net.weight)
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
J = X.Jinvp(a)
print(J)

X = pp.randn_SE3(6, requires_grad=True)
a = pp.randn_se3(6)
b = X.AdjT(a)
assert (X * b.Exp() - a.Exp() * X).abs().mean() < 1e-6

J = pp.Jinvp(X, a)
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
assert (X * b.Exp() - a.Exp() * X).abs().mean() < 1e-7

S = pp.randn_RxSO3(6)
s = pp.randn_rxso3(6)
s.Exp() * S

X = pp.randn_SE3(8, requires_grad=True, dtype=torch.double)
print(X.matrix())
print(X.translation())
assert hasattr(X.view(2,4,7), 'ltype')
X.lview(2,4)
print(X)


X = pp.randn_SE3(2, requires_grad=True, dtype=torch.double)
Y = pp.randn_SE3(2, requires_grad=True, dtype=torch.double)
from torch.overrides import get_overridable_functions
func_dict = get_overridable_functions()

Z = torch.cat([X,Y], dim=0)
assert isinstance(Z, pp.LieTensor)
assert isinstance(torch.stack([X,Y], dim=0), pp.LieTensor)

print(torch.stack([X,Y], dim=0))
Z1, Z2 = Z.split([2,2], dim=0)
assert isinstance(Z1, pp.LieTensor)
assert isinstance(Z2, pp.LieTensor)
assert not isinstance(torch.randn(4), pp.LieTensor)

a, b = Z.tensor_split(2)
assert isinstance(a, pp.LieTensor) and isinstance(b, pp.LieTensor)

x = pp.randn_SO3(4)
print(x)
x[0] = pp.randn_SO3(1)
print(x)
y = x.cuda()
print(x, y)
print(x.to("cuda"))

a = pp.randn_SE3()
y = a.sin()
print(type(a.sin().as_subclass(torch.Tensor))) # <class 'pypose.liegroup.group.groups.LieTensor'>
print(type(y))
print(y)

a = pp.randn_SE3(requires_grad=True)
b = a.sin()
print(type(b)) # <class 'pypose.liegroup.group.groups.LieTensor'>
c = torch.autograd.grad(b.sum(), a)
print(type(c[0])) #
print(type(a.tensor()))

a = pp.mat2SO3(torch.eye(3,3))
b = pp.randn_SO3(2,2)
print(a.shape, b.shape)
print((a*b).shape)


x = pp.randn_so3()
print(x)

X = pp.randn_SO3(3,2).cuda()
print(X)
X.identity_()
print(X)

euler = torch.randn(5,3).cuda()
X = pp.euler2SO3(euler)
print(X)

for i in range(128):
    x = torch.randn(4, 5, device="cuda:0")
    dim = torch.randint(0, 2, (1,)).item()
    assert torch.allclose(x.cumsum(dim=dim), pp.cumops(x, dim, lambda a, b : a + b), atol=1e-07)

for i in range(1, 1000):
    x = torch.randn(i, dtype=torch.float64)
    print(i, torch.allclose(x.cumsum(0), pp.cumops(x, 0, lambda a, b : a + b)))

x = pp.randn_SE3(2)
print(x, pp.cumprod(x, dim=0))

x = pp.randn_SE3(3)
print(x, x.cumprod(dim=0))
print(x)
x.cumprod_(dim=0)
print(x)

generator = torch.Generator()
x = pp.randn_so3(1, 2, sigma= 0.1, generator=generator, dtype=torch.float16)
print(x)

x = pp.randn_so3(2,2)
x.Jr()
pp.Jr(x)

x = pp.randn_SO3(2, device='cuda')
p = pp.randn_so3(2, device='cuda')
x.Jinvp(p)

from torch.autograd.functional import jacobian

def func(x):
    return x.Exp()

J = jacobian(func, p)
print("j:", J)

x = pp.randn_so3(2,1,2, requires_grad=True)
J = x.Jr()
print(x.shape, J.shape)
assert J.requires_grad is True

class PoseTransform(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.p = pp.Parameter(pp.randn_so3(2))

    def forward(self, x):
        return self.p.Exp() * x

model, inputs = PoseTransform(), pp.randn_SO3()
J = pp.optim.modjac(model, inputs, flatten=True)
print(J, J.shape)

LT = [pp.randn_SO3, pp.randn_so3, pp.randn_SE3, pp.randn_se3, \
      pp.randn_Sim3, pp.randn_sim3, pp.randn_RxSO3, pp.randn_rxso3]
import random
import torch.linalg
for lt in LT:
    x = lt(random.randint(1, 10), dtype=torch.float64, device='cuda', requires_grad=True)
    t = x.translation()
    r = x.rotation()
    s = x.scale()
    m = x.matrix()
    assert(r.device == t.device == s.device == x.device == m.device)
    assert(r.dtype == t.dtype == s.dtype == x.dtype == m.dtype)
    assert(r.lshape == t.shape[:-1] == s.shape[:-1] == x.lshape == m.shape[:-2])
    assert(r.requires_grad == t.requires_grad == s.requires_grad == x.requires_grad == m.requires_grad)
    assert(t.shape[-1]==3 and r.shape[-1]==4 and s.shape[-1]==1 and m.shape[-1] == m.shape[-2])
    assert(torch.all(torch.abs(torch.linalg.vector_norm(r, dim=1) - 1) < 1e-9))
    if m.shape[-1] == 4:
        assert(torch.all(m[:, 0:3, 3:4].view(-1, 3) == t))
    assert(torch.all(m[:, 0:3, 0:3] == s.view(-1, 1, 1) * r.matrix()))

print('Done')
