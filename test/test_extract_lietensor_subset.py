import pypose as pp
import torch
import random


def test_SO3(x):
    assert x.tensor().equal(x.rotation().tensor())
    assert torch.zeros(x.size()[:-1]+(3,)).equal(x.translation())
    assert torch.ones(x.size()[:-1]+(1,)).equal(x.scale())

def test_SE3(x):
    assert x.tensor()[..., 0:4].equal(x.rotation().tensor())
    assert x.tensor()[..., 4:6].equal(x.translation())
    assert torch.ones(x.size()[:-1]+(1,)).equal(x.scale())

def test_Sim3(x):
    assert x.tensor()[..., 0:4].equal(x.rotation().tensor())
    assert x.tensor()[..., 4:6].equal(x.translation())
    assert x.tensor()[..., 6:7].equal(x.scale())

def test_RxSO3(x):
    assert x.tensor()[..., 0:4].equal(x.rotation().tensor())
    assert torch.zeros(x.size()[:-1]+(3,)).equal(x.translation())
    assert x.tensor()[..., 4:5].equal(x.scale())

def test_so3(x):
    test_SO3(x.Exp())

def test_se3(x):
    test_SE3(x.Exp())

def test_sim3(x):
    test_Sim3(x.Exp())

def test_rxso3(x):
    test_RxSO3(x.Exp())


x = pp.randn_SO3(random.randint(1, 1000))
test_SO3(x)
x = pp.randn_SO3(random.randint(1, 10), random.randint(1, 10))
test_SO3(x)
x = pp.randn_SO3(random.randint(1, 10), random.randint(1, 10), random.randint(1, 10))
test_SO3(x)

x = pp.randn_so3(random.randint(1, 1000))
test_so3(x)
x = pp.randn_so3(random.randint(1, 10), random.randint(1, 10))
test_so3(x)
x = pp.randn_so3(random.randint(1, 10), random.randint(1, 10), random.randint(1, 10))
test_so3(x)

x = pp.randn_SE3(random.randint(1, 1000))
test_SE3(x)
x = pp.randn_SE3(random.randint(1, 10), random.randint(1, 10))
test_SE3(x)
x = pp.randn_SE3(random.randint(1, 10), random.randint(1, 10), random.randint(1, 10))
test_SE3(x)

x = pp.randn_se3(random.randint(1, 1000))
test_se3(x)
x = pp.randn_se3(random.randint(1, 10), random.randint(1, 10))
test_se3(x)
x = pp.randn_se3(random.randint(1, 10), random.randint(1, 10), random.randint(1, 10))
test_se3(x)

x = pp.randn_Sim3(random.randint(1, 1000))
test_Sim3(x)
x = pp.randn_Sim3(random.randint(1, 10), random.randint(1, 10))
test_Sim3(x)
x = pp.randn_Sim3(random.randint(1, 10), random.randint(1, 10), random.randint(1, 10))
test_Sim3(x)

x = pp.randn_sim3(random.randint(1, 1000))
test_sim3(x)
x = pp.randn_sim3(random.randint(1, 10), random.randint(1, 10))
test_sim3(x)
x = pp.randn_sim3(random.randint(1, 10), random.randint(1, 10), random.randint(1, 10))
test_sim3(x)

x = pp.randn_RxSO3(random.randint(1, 1000))
test_RxSO3(x)
x = pp.randn_RxSO3(random.randint(1, 10), random.randint(1, 10))
test_RxSO3(x)
x = pp.randn_RxSO3(random.randint(1, 10), random.randint(1, 10), random.randint(1, 10))
test_RxSO3(x)

x = pp.randn_rxso3(random.randint(1, 1000))
test_rxso3(x)
x = pp.randn_rxso3(random.randint(1, 10), random.randint(1, 10))
test_rxso3(x)
x = pp.randn_rxso3(random.randint(1, 10), random.randint(1, 10), random.randint(1, 10))
test_rxso3(x)
