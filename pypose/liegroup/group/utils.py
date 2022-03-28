import torch
import functools
from .groups import  LieGroup
from .groups import SE3_type, se3_type
from .groups import SO3_type, so3_type
from .groups import Sim3_type, sim3_type
from .groups import RxSO3_type, rxso3_type


SO3 = functools.partial(LieGroup, gtype=SO3_type)
so3 = functools.partial(LieGroup, gtype=so3_type)
SE3 = functools.partial(LieGroup, gtype=SE3_type)
se3 = functools.partial(LieGroup, gtype=se3_type)
Sim3 = functools.partial(LieGroup, gtype=Sim3_type)
sim3 = functools.partial(LieGroup, gtype=sim3_type)
RxSO3 = functools.partial(LieGroup, gtype=RxSO3_type)
rxso3 = functools.partial(LieGroup, gtype=rxso3_type)


def randn_like(liegroup, sigma=1, **kwargs):
    return liegroup.gtype.randn_like(*liegroup.gshape, sigma=sigma, **kwargs)


def randn_so3(*args, sigma=1, requires_grad=False, **kwargs):
    return so3_type.randn(*args, sigma=sigma, requires_grad=requires_grad, **kwargs)


def randn_SO3(*args, sigma=1, requires_grad=False, **kwargs):
    return SO3_type.randn(*args, sigma=sigma, requires_grad=requires_grad, **kwargs)


def randn_se3(*args, sigma=1, requires_grad=False, **kwargs):
    return se3_type.randn(*args, sigma=sigma, requires_grad=requires_grad, **kwargs)


def randn_SE3(*args, sigma=1, requires_grad=False, **kwargs):
    return SE3_type.randn(*args, sigma=sigma, requires_grad=requires_grad, **kwargs)


def randn_sim3(*args, sigma=1, requires_grad=False, **kwargs):
    return sim3_type.randn(*args, sigma=sigma, requires_grad=requires_grad, **kwargs)


def randn_Sim3(*args, sigma=1, requires_grad=False, **kwargs):
    return Sim3_type.randn(*args, sigma=sigma, requires_grad=requires_grad, **kwargs)


def randn_rxso3(*args, sigma=1, requires_grad=False, **kwargs):
    return rxso3_type.randn(*args, sigma=sigma, requires_grad=requires_grad, **kwargs)


def randn_RxSO3(*args, sigma=1, requires_grad=False, **kwargs):
    return RxSO3_type.randn(*args, sigma=sigma, requires_grad=requires_grad, **kwargs)


def identity_like(liegroup, **kwargs):
    return liegroup.gtype.identity_like(*liegroup.gshape, **kwargs)


def identity_SO3(*args, **kwargs):
    return SO3_type.identity(*args, **kwargs)


def identity_so3(*args, **kwargs):
    return so3_type.identity(*args, **kwargs)


def identity_SE3(*args, **kwargs):
    return SE3_type.identity(*args, **kwargs)


def identity_se3(*args, **kwargs):
    return se3_type.identity(*args, **kwargs)


def identity_sim3(*args, **kwargs):
    return sim3_type.identity(*args, **kwargs)


def identity_Sim3(*args, **kwargs):
    return Sim3_type.identity(*args, **kwargs)


def identity_rxso3(*args, **kwargs):
    return rxso3_type.identity(*args, **kwargs)


def identity_RxSO3(*args, **kwargs):
    return RxSO3_type.identity(*args, **kwargs)


def assert_gtype(func):
    def checker(*args, **kwargs):
        assert isinstance(args[0], LieGroup), "Invalid LieGroup Type."
        out = func(*args, **kwargs)
        return out
    return checker


@assert_gtype
def Exp(x):
    return x.Exp()


@assert_gtype
def Log(x):
    return x.Log()


@assert_gtype
def Inv(x):
    return x.Inv()


@assert_gtype
def Mul(x, y):
    return x * y


@assert_gtype
def Retr(X, a):
    return X.Retr(a)


@assert_gtype
def Act(X, p):
    return X.Act(p)


@assert_gtype
def Adj(X, a):
    return X.Adj(a)


@assert_gtype
def AdjT(X, a):
    return X.AdjT(a)


@assert_gtype
def Jinv(X, a):
    return X.Jinv(a)
