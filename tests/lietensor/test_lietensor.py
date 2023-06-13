import time
import copy
import torch
import random
import warnings
import torch.linalg
import pypose as pp
from torch import nn
import pypose.optim.solver as ppos
import pypose.optim.kernel as ppok
import pypose.optim.corrector as ppoc
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_lietensor():

    a = torch.randn(3,3)*0.1
    x = pp.so3(a)
    y = pp.randn_so3(3, sigma=0.1, dtype=torch.float64, requires_grad=True, device=device)
    a = pp.randn_se3(3, sigma=0.1, requires_grad=True)
    b = pp.SE3_type.randn(3, sigma=0.1, requires_grad=True)

    assert y.is_leaf and x.is_leaf and a.is_leaf and b.is_leaf

    (pp.Log(y.Exp())**2).sin().sum().backward()

    assert not torch.any(torch.isnan(y.grad))

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
    net = SO3Layer(n).to(device)
    weight = copy.deepcopy(net.weight)
    optimizer = torch.optim.SGD(net.parameters(), lr = 0.1, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,15], gamma=0.1)
    for i in range(epoch):
        optimizer.zero_grad()
        inputs = pp.randn_SO3(n, device=device)
        outputs = net(inputs)
        loss = outputs.abs().sum()
        loss.backward()
        optimizer.step()
        scheduler.step()

    assert not torch.allclose(weight, net.weight)

    SO3I = pp.identity_SO3(1, 3, device=device, dtype=torch.float64)
    so3I = pp.identity_so3(2, 1, device=device, requires_grad=True)

    SE3I = pp.identity_SE3(3, device=device, dtype=torch.float64)
    se3I = pp.identity_se3(2)

    idl1 = pp.identity_like(x)
    idl2 = pp.identity_like(a, requires_grad=True)

    rdl1 = pp.randn_like(x)
    rdl2 = pp.randn_like(a, sigma=0.1, requires_grad=True, device=device)

    b.Inv()

    assert pp.is_lietensor(pp.randn_so3(2).Inv())

    inv = pp.randn_SE3(4)
    torch.testing.assert_close(inv.Inv().Log(), inv.Log().Inv())

    inv = pp.randn_SO3(4)
    torch.testing.assert_close(inv.Inv().Log(), inv.Log().Inv())

    inv = pp.randn_RxSO3(4)
    torch.testing.assert_close(inv.Inv().Log(), inv.Log().Inv())

    inv = pp.randn_Sim3(4)
    torch.testing.assert_close(inv.Inv().Log(), inv.Log().Inv())

    a = pp.randn_SE3(1,5)
    b = pp.randn_SE3(5,1)
    c = pp.randn_SO3(1,5)

    e = a * b

    g = pp.randn_so3(1,5)
    g * 0.1

    pp.Mul(a, b)

    points = torch.randn(2,5,3)

    pt = a * points

    a = pp.randn_so3(3)
    I = pp.identity_SO3(3)
    t = I.Retr(a)

    I = pp.identity_SE3(3)

    p = I.Act(a)

    X = pp.randn_SE3(8, requires_grad=True, dtype=torch.double)
    a = pp.randn_se3(8, dtype=torch.double)
    b = X.Adj(a)
    torch.testing.assert_close(b.Exp() * X , X * a.Exp())
    J = X.Jinvp(a)

    X = pp.randn_SE3(6, requires_grad=True)
    a = pp.randn_se3(6)
    b = X.AdjT(a)
    torch.testing.assert_close(X * b.Exp(), a.Exp() * X)

    J = pp.Jinvp(X, a)

    S = pp.randn_Sim3(4)
    S.Log().Exp()

    s = pp.randn_sim3(4)
    s.Exp().Log()

    X = pp.randn_Sim3(8, requires_grad=True, dtype=torch.double)
    a = pp.randn_sim3(8, dtype=torch.double)
    b = X.Adj(a)
    torch.testing.assert_close(b.Exp() * X, X * a.Exp())

    X = pp.randn_Sim3(6, requires_grad=True, dtype=torch.double)
    a = pp.randn_sim3(6, dtype=torch.double)
    b = X.AdjT(a)
    torch.testing.assert_close(X * b.Exp(), a.Exp() * X)

    S = pp.randn_RxSO3(6)
    s = pp.randn_rxso3(6)
    s.Exp() * S

    X = pp.randn_SE3(8, requires_grad=True, dtype=torch.double)
    assert hasattr(X.view(2,4,7), 'ltype')
    X.lview(2,4)

    X = pp.randn_SE3(2, requires_grad=True, dtype=torch.double)
    Y = pp.randn_SE3(2, requires_grad=True, dtype=torch.double)
    from torch.overrides import get_overridable_functions
    func_dict = get_overridable_functions()

    Z = torch.cat([X,Y], dim=0)
    assert isinstance(Z, pp.LieTensor)
    assert isinstance(torch.stack([X,Y], dim=0), pp.LieTensor)

    Z1, Z2 = Z.split([2,2], dim=0)
    assert isinstance(Z1, pp.LieTensor)
    assert isinstance(Z2, pp.LieTensor)
    assert not isinstance(torch.randn(4), pp.LieTensor)

    a, b = Z.tensor_split(2)
    assert isinstance(a, pp.LieTensor) and isinstance(b, pp.LieTensor)

    x = pp.randn_SO3(4)
    x[0] = pp.randn_SO3(1)
    y = x.to(device)


    a = pp.randn_SE3()
    y = a.sin()
    a = pp.randn_SE3(requires_grad=True)
    b = a.sin()
    c = torch.autograd.grad(b.sum(), a)

    a = pp.mat2SO3(torch.eye(3,3))
    b = pp.randn_SO3(2,2)
    x = pp.randn_so3()

    X = pp.randn_SO3(3,2).to(device)
    X.identity_()

    euler = torch.randn(5,3).to(device)
    X = pp.euler2SO3(euler)

    for i in range(128):
        x = torch.randn(4, 5, device=device)
        dim = torch.randint(0, 2, (1,)).item()
        torch.testing.assert_close(x.cumsum(dim=dim), pp.cumops(x, dim, lambda a, b : a + b))

    for i in range(1, 1000):
        x = torch.randn(i, dtype=torch.float64)
        torch.testing.assert_close(x.cumsum(0), pp.cumops(x, 0, lambda a, b : a + b))

    x = pp.randn_SE3(2)

    x = pp.randn_SE3(3)
    y = copy.deepcopy(x)
    x.cumprod_(dim=0)
    assert not torch.allclose(x, y)

    generator = torch.Generator()
    x = pp.randn_so3(1, 2, sigma= 0.1, generator=generator, dtype=torch.float16) # not half

    x = pp.randn_so3(2,2)
    x.Jr()
    pp.Jr(x)

    x = pp.randn_SO3(2, device=device)
    p = pp.randn_so3(2, device=device)
    x.Jinvp(p)

    from torch.autograd.functional import jacobian

    def func(x):
        return x.Exp().tensor()

    J = jacobian(func, p)

    x = pp.randn_so3(2,1,2, requires_grad=True)
    J = x.Jr()
    assert J.requires_grad is True

    class PoseTransform(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.p = pp.Parameter(pp.randn_so3(2))

        def forward(self, x):
            return (self.p.Exp() * x).tensor()

    model, input = PoseTransform(), pp.randn_SO3()
    J = pp.optim.functional.modjac(model, input=input, flatten=True)

    LT = [pp.randn_SO3, pp.randn_so3, pp.randn_SE3, pp.randn_se3, \
        pp.randn_Sim3, pp.randn_sim3, pp.randn_RxSO3, pp.randn_rxso3]

    warnings.filterwarnings("ignore", message="Instance has no translation.")
    warnings.filterwarnings("ignore", message="Instance has no scale.")
    for lt in LT:
        x = lt(random.randint(1, 10), dtype=torch.float64, device=device, requires_grad=True)
        t = x.translation()
        r = x.rotation()
        s = x.scale()
        m = x.matrix()
        xx = pp.randn_like(x)
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


if __name__ == '__main__':
    test_lietensor()
