import torch
from torch import nn
from .broadcasting import broadcast_inputs
from .group_ops import exp, log, inv, mul, adj
from .group_ops import adjT, jinv, act3, act4, toMatrix


class GroupType:
    '''Lie Group Type Base Class'''
    def __init__(self, groud,  dimension, embedding, manifold):
        self._group     = groud     # Group ID
        self._dimension = dimension # Data dimension
        self._embedding = embedding # Embedding dimension
        self._manifold  = manifold  # Manifold dimension

    @property
    def group(self):
        return self._group

    @property
    def dimension(self):
        return self._dimension

    @property
    def embedding(self):
        return self._embedding

    @property
    def manifold(self):
        return self._manifold

    @property
    def on_manifold(self):
        return self.dimension == self.manifold

    def Log(self, X):
        if self.on_manifold:
            raise AttributeError("Manifold Type has no Log attribute")
        raise NotImplementedError("Instance has no Log attribute.")

    def Exp(self, x):
        if not self.on_manifold:
            raise AttributeError("Embedding Type has no Exp attribute")
        raise NotImplementedError("Instance has no Exp attribute.")

    def Inv(self, x):
        if self.on_manifold:
            return LieGroup(-x, gtype=x.gtype, requires_grad=x.requires_grad)
        out = self.__op__(self.group, inv, x)
        return LieGroup(out, gtype=x.gtype, requires_grad=x.requires_grad)

    def Act(self, x, p):
        """ action on a points tensor(*, 3[4]) (homogeneous)"""
        assert not self.on_manifold and isinstance(p, torch.Tensor)
        assert p.shape[-1]==3 or p.shape==4, "Invalid Tensor Dimension"
        act = act3 if p.shape[-1]==3 else act4
        return self.__op__(self.group, act, x, p)

    def Mul(self, x, y):
        # Transform on transform
        if not self.on_manifold and isinstance(y, LieGroup) and not y.gtype.on_manifold:
            out = self.__op__(self.group, mul, x, y)
            return LieGroup(out, gtype=x.gtype, requires_grad=x.requires_grad)
        # Transform on points
        if not self.on_manifold and isinstance(y, torch.Tensor):
            return self.Act(x, y)
        # scalar * manifold
        if self.on_manifold:
            if isinstance(y, torch.Tensor):
                assert y.dim()==0 or y.shape[-1] ==1, "Tensor Dimension Invalid"
            return torch.mul(x, y)
        raise NotImplementedError('Invalid __mul__ operation')

    def Retr(self, X, a):
        if self.on_manifold:
            raise AttributeError("Gtype has no Retr attribute")
        return a.Exp() * X

    def Adj(self, X, a):
        ''' X * Exp(a) = Exp(Adj) * X '''
        if self.on_manifold:
            raise AttributeError("Gtype has no Adj attribute")
        assert not X.gtype.on_manifold and a.gtype.on_manifold
        assert X.gtype.group == a.gtype.group
        out = self.__op__(self.group, adj, X, a)
        return LieGroup(out, gtype=a.gtype, requires_grad=a.requires_grad or X.requires_grad)

    def AdjT(self, X, a): # It seems that only works for SO3, but not SE3
        ''' Exp(a) * X = X * Exp(AdjT) '''
        if self.on_manifold:
            raise AttributeError("Gtype has no AdjT attribute")
        assert not X.gtype.on_manifold and a.gtype.on_manifold, "Gtype Invalid"
        assert X.gtype.group == a.gtype.group, "Gtype Invalid"
        out = self.__op__(self.group, adjT, X, a)
        return LieGroup(out, gtype=a.gtype, requires_grad=a.requires_grad or X.requires_grad)

    def Jinv(self, X, a):
        if self.on_manifold:
            raise AttributeError("Gtype has no Jinv attribute")
        return self.__op__(self.group, jinv, X, a)

    @classmethod
    def identity(cls, *args, **kwargs):
        raise NotImplementedError("Instance has no identity.")

    @classmethod
    def identity_like(cls, *args, **kwargs):
        return cls.identity(*args, **kwargs)

    def randn_like(self, *args, sigma=1, **kwargs):
        return self.randn(*args, sigma=1, **kwargs)

    def randn(self, *args, sigma=1., **kwargs):
        return sigma * torch.randn(*(list(args)+[self.manifold]), **kwargs)

    @classmethod
    def __op__(cls, group, op, x, y=None):
        inputs, out_shape = broadcast_inputs(x, y)
        out = op.apply(group, *inputs)
        return out.view(out_shape + (-1,))


class SO3Type(GroupType):
    def __init__(self):
        super().__init__(1, 4, 4, 3)

    def Log(self, X):
        x = self.__op__(self.group, log, X)
        return LieGroup(x, gtype=so3_type, requires_grad=X.requires_grad)

    @classmethod
    def identity(cls, *args, **kwargs):
        data = torch.tensor([0., 0., 0., 1.], **kwargs)
        return LieGroup(data.expand(args+(-1,)),
                gtype=SO3_type, requires_grad=data.requires_grad)

    def randn(self, *args, sigma=1, requires_grad=False, **kwargs):
        data = so3_type.Exp(so3_type.randn(*args, sigma=sigma, **kwargs)).detach()
        return LieGroup(data, gtype=SO3_type).requires_grad_(requires_grad)


class so3Type(GroupType):
    def __init__(self):
        super().__init__(1, 3, 4, 3)

    def Exp(self, x):
        X = self.__op__(self.group, exp, x)
        return LieGroup(X, gtype=SO3_type, requires_grad=x.requires_grad)

    @classmethod
    def identity(cls, *args, **kwargs):
        return SO3_type.Log(SO3_type.identity(*args, **kwargs))

    def randn(self, *args, sigma=1, requires_grad=False, **kwargs):
        data = super().randn(*args, sigma=sigma, **kwargs).detach()
        return LieGroup(data, gtype=so3_type).requires_grad_(requires_grad)


class SE3Type(GroupType):
    def __init__(self):
        super().__init__(3, 7, 7, 6)

    def Log(self, X):
        x = self.__op__(self.group, log, X)
        return LieGroup(x, gtype=se3_type, requires_grad=X.requires_grad)

    @classmethod
    def identity(cls, *args, **kwargs):
        data = torch.tensor([0., 0., 0., 0., 0., 0., 1.], **kwargs)
        return LieGroup(data.expand(args+(-1,)),
                gtype=SE3_type, requires_grad=data.requires_grad)

    def randn(self, *args, sigma=1, requires_grad=False, **kwargs):
        data = se3_type.Exp(se3_type.randn(*args, sigma=sigma, **kwargs)).detach()
        return LieGroup(data, gtype=SE3_type).requires_grad_(requires_grad)


class se3Type(GroupType):
    def __init__(self):
        super().__init__(3, 6, 7, 6)

    def Exp(self, x):
        X = self.__op__(self.group, exp, x)
        return LieGroup(X, gtype=SE3_type, requires_grad=x.requires_grad)

    @classmethod
    def identity(cls, *args, **kwargs):
        return SE3_type.Log(SE3_type.identity(*args, **kwargs))

    def randn(self, *args, sigma=1, requires_grad=False, **kwargs):
        data = super().randn(*args, sigma=sigma, **kwargs).detach()
        return LieGroup(data, gtype=se3_type).requires_grad_(requires_grad)


SO3_type, so3_type = SO3Type(), so3Type()
SE3_type, se3_type = SE3Type(), se3Type()


class LieGroup(torch.Tensor):
    """ Lie Group """
    from torch._C import _disabled_torch_function_impl
    __torch_function__ = _disabled_torch_function_impl

    def __init__(self, data, gtype=None, **kwargs):
        assert data.shape[-1] == gtype.dimension, 'Dimension Invalid.'
        self.gtype = gtype

    def __new__(cls, data=None, **kwargs):
        if data is None:
            data = torch.tensor([])
        return torch.Tensor.as_subclass(data, LieGroup) 

    def __repr__(self):
        return self.gtype.__class__.__name__ + " Group:\n" + super().__repr__()

    @property
    def gshape(self):
        return self.shape[:-1]
    
    def tensor(self):
        return self.data

    def Exp(self):
        return self.gtype.Exp(self)

    def Log(self):
        return self.gtype.Log(self)

    def Inv(self):
        return self.gtype.Inv(self)

    def Act(self, p):
        return self.gtype.Act(self, p)

    def __mul__(self, other):
        return self.gtype.Mul(self, other)

    def Retr(self, a):
        return self.gtype.Retr(self, a)

    def Adj(self, a):
        return self.gtype.Adj(self, a)

    def AdjT(self, a):
        return self.gtype.AdjT(self, a)

    def Jinv(self, a):
        return self.gtype.Jinv(self, a)

    def matrix(self):
        """ convert element to 4x4 matrix """
        I = torch.eye(4, dtype=self.dtype, device=self.device)
        I = I.view([1] * (len(self.data.shape) - 1) + [4, 4])
        return self.__class__(self.data[...,None,:]).act(I).transpose(-1,-2)

    def translation(self):
        """ extract translation component """
        p = torch.as_tensor([0.0, 0.0, 0.0, 1.0], dtype=self.dtype, device=self.device)
        p = p.view([1] * (len(self.data.shape) - 1) + [4,])
        return self.apply_op(Act4, self.data, p)


class Parameter(LieGroup, nn.Parameter):
    def __new__(cls, data=None, gtype=None, requires_grad=True):
        if data is None:
            data = torch.tensor([])
        return LieGroup._make_subclass(cls, data, requires_grad)
