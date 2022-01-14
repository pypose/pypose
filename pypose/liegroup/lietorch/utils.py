import torch
from .groups import  LieGroup, SE3_type, se3_type, SO3_type, so3_type


def SO3(data, **kwargs):
    assert data.shape[-1] == SO3_type.dimension, 'Dimension Invalid.'
    return LieGroup(data, gtype=SO3_type, **kwargs)


def so3(data, **kwargs):
    assert data.shape[-1] == so3_type.dimension, 'Dimension Invalid.'
    return LieGroup(data, gtype=so3_type, **kwargs)


def SE3(data, **kwargs):
    assert data.shape[-1] == SE3_type.dimension, 'Dimension Invalid.'
    return LieGroup(data, gtype=SE3_type, **kwargs)


def se3(data, **kwargs):
    assert data.shape[-1] == se3_type.dimension, 'Dimension Invalid.'
    return LieGroup(data, gtype=se3_type, **kwargs)


def randn_so3(*args, sigma=1, requires_grad=False, **kwargs):
    return so3_type.randn(*args, sigma=sigma, requires_grad=requires_grad, **kwargs)


def randn_SO3(*args, sigma=1, requires_grad=False, **kwargs):
    return SO3_type.randn(*args, sigma=sigma, requires_grad=requires_grad, **kwargs)


def randn_se3(*args, sigma=1, requires_grad=False, **kwargs):
    return se3_type.randn(*args, sigma=sigma, requires_grad=requires_grad, **kwargs)


def randn_SE3(*args, sigma=1, requires_grad=False, **kwargs):
    return SE3_type.randn(*args, sigma=sigma, requires_grad=requires_grad, **kwargs)


def identity_SO3(*args, **kwargs):
    return SO3_type.identity(*args, **kwargs)


def identity_so3(*args, **kwargs):
    return so3_type.identity(*args, **kwargs)


def identity_SE3(*args, **kwargs):
    return SE3_type.identity(*args, **kwargs)


def identity_se3(*args, **kwargs):
    return se3_type.identity(*args, **kwargs)


def Exp(x):
    assert isinstance(x, LieGroup), "Not a LieGroup Instance"
    return x.Exp()


def Log(x):
    assert isinstance(x, LieGroup), "Not a LieGroup Instance"
    return x.Log()
