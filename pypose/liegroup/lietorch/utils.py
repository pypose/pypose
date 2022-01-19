import torch
import functools
from .groups import  LieGroup, SE3_type, se3_type, SO3_type, so3_type


SO3 = functools.partial(LieGroup, gtype=SO3_type)
so3 = functools.partial(LieGroup, gtype=so3_type)
SE3 = functools.partial(LieGroup, gtype=SE3_type)
se3 = functools.partial(LieGroup, gtype=se3_type)


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


def Exp(x):
    return x.Exp()


def Log(x):
    return x.Log()


def Inv(x):
    return x.Inv()


def Mul(x, y):
    return x * y


def Retr(X, a):
    return X.Retr(a)


def Act(X, p):
    return X.Act(p)
