import torch
from torch import nn
from .basics import vec2skew
import collections, numbers, warnings
from .operation import broadcast_inputs
from torch.utils._pytree import tree_map, tree_flatten
from .operation import SO3_Log, SE3_Log, RxSO3_Log, Sim3_Log
from .operation import so3_Exp, se3_Exp, rxso3_Exp, sim3_Exp
from .operation import SO3_Act, SE3_Act, RxSO3_Act, Sim3_Act
from .operation import SO3_Mul, SE3_Mul, RxSO3_Mul, Sim3_Mul
from .operation import SO3_Inv, SE3_Inv, RxSO3_Inv, Sim3_Inv
from .operation import SO3_Act4, SE3_Act4, RxSO3_Act4, Sim3_Act4
from .operation import SO3_AdjXa, SE3_AdjXa, RxSO3_AdjXa, Sim3_AdjXa
from .operation import SO3_AdjTXa, SE3_AdjTXa, RxSO3_AdjTXa, Sim3_AdjTXa
from .operation import so3_Jl_inv, se3_Jl_inv, rxso3_Jl_inv, sim3_Jl_inv
from ..basics import pm, cumops_, cummul_, cumprod_, cumops, cummul, cumprod
from torch.nn.modules.utils import _single, _pair, _triple, _quadruple, _ntuple


HANDLED_FUNCTIONS = ['__getitem__', '__setitem__', 'cpu', 'cuda', 'float', 'double',
                     'to', 'detach', 'view', 'view_as', 'squeeze', 'unsqueeze', 'cat',
                     'stack', 'split', 'hsplit', 'dsplit', 'vsplit', 'tensor_split',
                     'chunk', 'concat', 'column_stack', 'dstack', 'vstack', 'hstack',
                     'index_select', 'masked_select', 'movedim', 'moveaxis', 'narrow',
                     'permute', 'reshape', 'row_stack', 'scatter', 'scatter_add', 'clone',
                     'swapaxes', 'swapdims', 'take', 'take_along_dim', 'tile', 'copy',
                     'transpose', 'unbind', 'gather', 'repeat', 'expand', 'expand_as',
                     'index_select', 'masked_select', 'index_copy', 'index_copy_',
                     'select', 'select_scatter', 'index_put','index_put_', 'copy_']

class LieType:
    '''LieTensor Type Base Class'''
    def __init__(self, dimension, embedding, manifold):
        self._dimension = torch.Size([dimension]) # Data dimension
        self._embedding = torch.Size([embedding]) # Embedding dimension
        self._manifold  = torch.Size([manifold])  # Manifold dimension

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

    def add_(self, input, other):
        if self.on_manifold:
            other1 = torch.Tensor.as_subclass(input, torch.Tensor)
            other2 = torch.Tensor.as_subclass(other, torch.Tensor)
            return input.copy_(other1 + other2[..., :self.manifold[0]])
        raise NotImplementedError("Instance has no add_ attribute.")

    def Log(self, X):
        if self.on_manifold:
            raise AttributeError("Lie Algebra has no Log attribute")
        raise NotImplementedError("Instance has no Log attribute.")

    def Exp(self, x):
        if not self.on_manifold:
            raise AttributeError("Lie Group has no Exp attribute")
        raise NotImplementedError("Instance has no Exp attribute.")

    def Inv(self, x):
        if self.on_manifold:
            return - x
        raise NotImplementedError("Instance has no Inv attribute.")

    def Act(self, X, p):
        """ action on a points tensor(*, 3[4]) (homogeneous)"""
        if not self.on_manifold:
            raise AttributeError("Lie Group has no Act attribute")
        raise NotImplementedError("Instance has no Act attribute.")

    def Mul(self, X, Y):
        if not self.on_manifold:
            raise AttributeError("Lie Group has no Mul attribute")
        raise NotImplementedError("Instance has no Mul attribute.")

    def Retr(self, X, a):
        if self.on_manifold:
            raise AttributeError("Has no Retr attribute")
        return a.Exp() * X

    def Adj(self, X, a):
        ''' X * Exp(a) = Exp(Adj) * X '''
        if not self.on_manifold:
            raise AttributeError("Lie Group has no Adj attribute")
        raise NotImplementedError("Instance has no Adj attribute.")

    def AdjT(self, X, a):
        ''' Exp(a) * X = X * Exp(AdjT) '''
        if not self.on_manifold:
            raise AttributeError("Lie Group has no AdjT attribute")
        raise NotImplementedError("Instance has no AdjT attribute.")

    def Jinvp(self, X, p):
        if not self.on_manifold:
            raise AttributeError("Lie Group has no Jinvp attribute")
        raise NotImplementedError("Instance has no Jinvp attribute.")

    def matrix(self, input):
        """ To 4x4 matrix """
        X = input.Exp() if self.on_manifold else input
        I = torch.eye(4, dtype=X.dtype, device=X.device)
        I = I.view([1] * (X.dim() - 1) + [4, 4])
        return X.unsqueeze(-2).Act(I).transpose(-1,-2)

    def rotation(self, input):
        raise NotImplementedError("Rotation is not implemented for the instance.")

    def translation(self, input):
        warnings.warn("Instance has no translation. Zero vector(s) is returned.")
        return torch.zeros(input.lshape + (3,), dtype=input.dtype, device=input.device,
            requires_grad=input.requires_grad)

    def scale(self, input):
        warnings.warn("Instance has no scale. Scalar one(s) is returned.")
        return torch.ones(input.lshape + (1,), dtype=input.dtype, device=input.device,
            requires_grad=input.requires_grad)

    @classmethod
    def to_tuple(cls, input):
        out = tuple()
        for i in input:
            if not isinstance(i, collections.abc.Iterable):
                out += (i,)
            else:
                out += tuple(i)
        return out

    @classmethod
    def identity(cls, *args, **kwargs):
        raise NotImplementedError("Instance has no identity.")

    @classmethod
    def identity_like(cls, *args, **kwargs):
        return cls.identity(*args, **kwargs)

    def randn_like(self, *args, sigma=1.0, **kwargs):
        return self.randn(*args, sigma=sigma, **kwargs)

    def randn(self, *args, **kwargs):
        raise NotImplementedError("randn not implemented yet")

    @classmethod
    def cumops(self, X, dim, ops):
        return cumops(X, dim, ops)

    @classmethod
    def cummul(self, X, dim):
        return cummul(X, dim)

    @classmethod
    def cumprod(self, X, dim, left = True):
        return cumprod(X, dim, left)

    @classmethod
    def cumops_(self, X, dim, ops):
        return cumops_(X, dim, ops)

    @classmethod
    def cummul_(self, X, dim):
        return cummul_(X, dim)

    @classmethod
    def cumprod_(self, X, dim):
        return cumprod_(X, dim)


class SO3Type(LieType):
    def __init__(self):
        super().__init__(4, 4, 3)

    def Log(self, X):
        X = X.tensor() if hasattr(X, 'ltype') else X
        x = SO3_Log.apply(X)
        return LieTensor(x, ltype=so3_type)
    
    def Act(self, X, p):
        assert not self.on_manifold and isinstance(p, torch.Tensor)
        assert p.shape[-1]==3 or p.shape[-1]==4, "Invalid Tensor Dimension"
        X = X.tensor() if hasattr(X, 'ltype') else X
        input, out_shape = broadcast_inputs(X, p)
        if p.shape[-1]==3:
            out = SO3_Act.apply(*input)
        else:
            out = SO3_Act4.apply(*input)
        dim = -1 if out.nelement() != 0 else X.shape[-1]
        return out.view(out_shape + (dim,))

    def Mul(self, X, Y):
        # Transform on transform
        X = X.tensor() if hasattr(X, 'ltype') else X
        if not self.on_manifold and isinstance(Y, LieTensor) and not Y.ltype.on_manifold:
            Y = Y.tensor() if hasattr(Y, 'ltype') else Y
            input, out_shape = broadcast_inputs(X, Y)
            out = SO3_Mul.apply(*input)
            dim = -1 if out.nelement() != 0 else X.shape[-1]
            out = out.view(out_shape + (dim,))
            return LieTensor(out, ltype=SO3_type)
        # Transform on points
        if not self.on_manifold and isinstance(Y, torch.Tensor):
            return self.Act(X, Y)
        # (scalar or tensor) * manifold
        if self.on_manifold:
            return LieTensor(torch.mul(X, Y), ltype=SO3_type)
        raise NotImplementedError('Invalid __mul__ operation')
    
    def Inv(self, X):
        X = X.tensor() if hasattr(X, 'ltype') else X
        out = SO3_Inv.apply(X)
        return LieTensor(out, ltype=SO3_type)
    
    def Adj(self, X, a):
        X = X.tensor() if hasattr(X, 'ltype') else X
        a = a.tensor() if hasattr(a, 'ltype') else a
        input, out_shape = broadcast_inputs(X, a)
        out = SO3_AdjXa.apply(*input)
        dim = -1 if out.nelement() != 0 else X.shape[-1]
        out = out.view(out_shape + (dim,))
        return LieTensor(out, ltype=so3_type)

    def AdjT(self, X, a):
        X = X.tensor() if hasattr(X, 'ltype') else X
        a = a.tensor() if hasattr(a, 'ltype') else a
        input, out_shape = broadcast_inputs(X, a)
        out = SO3_AdjTXa.apply(*input)
        dim = -1 if out.nelement() != 0 else X.shape[-1]
        out = out.view(out_shape + (dim,))
        return LieTensor(out, ltype=so3_type)

    def Jinvp(self, X, a):
        X = X.tensor() if hasattr(X, 'ltype') else X
        a = a.tensor() if hasattr(a, 'ltype') else a
        (X, a), out_shape = broadcast_inputs(X, a)
        out = (so3_Jl_inv(SO3_Log.apply(X)) @ a.unsqueeze(-1)).squeeze(-1)
        dim = -1 if out.nelement() != 0 else X.shape[-1]
        out = out.view(out_shape + (dim,))
        return LieTensor(out, ltype=so3_type)

    @classmethod
    def identity(cls, *size, **kwargs):
        data = torch.tensor([0., 0., 0., 1.], **kwargs)
        return LieTensor(data.repeat(size+(1,)), ltype=SO3_type)

    def randn(self, *size, sigma=1.0, requires_grad=False, **kwargs):
        data = so3_type.Exp(so3_type.randn(*size, sigma=sigma, **kwargs)).detach()
        return LieTensor(data, ltype=SO3_type).requires_grad_(requires_grad)

    @classmethod
    def add_(cls, input, other):
        return input.copy_(LieTensor(other[..., :3], ltype=so3_type).Exp() * input)

    def matrix(self, input):
        """ To 3x3 matrix """
        I = torch.eye(3, dtype=input.dtype, device=input.device)
        I = I.view([1] * (input.dim() - 1) + [3, 3])
        return input.unsqueeze(-2).Act(I).transpose(-1,-2)

    def rotation(self, input):
        return input

    def identity_(self, X):
        X.fill_(0)
        X.index_fill_(dim=-1, index=torch.tensor([-1], device=X.device), value=1)
        return X

    def Jr(self, X):
        """
        Right jacobian of SO(3)
        """
        return X.Log().Jr()


class so3Type(LieType):
    def __init__(self):
        super().__init__(3, 4, 3)

    def Exp(self, x):
        x = x.tensor() if hasattr(x, 'ltype') else x
        X = so3_Exp.apply(x)
        return LieTensor(X, ltype=SO3_type)

    def Mul(self, X, Y):
        X = X.tensor() if hasattr(X, 'ltype') else X
        # (scalar or tensor) * manifold
        if self.on_manifold:
            return LieTensor(torch.mul(X, Y), ltype=so3_type)
        raise NotImplementedError('Invalid __mul__ operation')

    @classmethod
    def identity(cls, *size, **kwargs):
        return SO3_type.Log(SO3_type.identity(*size, **kwargs))

    def randn(self, *size, sigma=1.0, requires_grad=False, **kwargs):
        assert isinstance(sigma, numbers.Number), 'Only accepts sigma as a single number'
        size = self.to_tuple(size)
        data = torch.randn(*(size + torch.Size([3])), **kwargs)
        dist = data.norm(dim=-1, keepdim=True)
        theta = sigma * torch.randn(*(size + torch.Size([1])), **kwargs)
        return LieTensor(data / dist * theta, ltype=so3_type).requires_grad_(requires_grad)

    def matrix(self, input):
        """ To 3x3 matrix """
        X = input.Exp()
        I = torch.eye(3, dtype=X.dtype, device=X.device)
        I = I.view([1] * (X.dim() - 1) + [3, 3])
        return X.unsqueeze(-2).Act(I).transpose(-1,-2)

    def rotation(self, input):
        return input.Exp().rotation()

    def Jr(self, x):
        """
        Right jacobian of so(3)
        """
        K = vec2skew(x)
        theta = torch.linalg.norm(x, dim=-1, keepdim=True).unsqueeze(-1)
        I = torch.eye(3, device=x.device, dtype=x.dtype).expand(x.lshape+(3, 3))
        Jr = I - (1-theta.cos())/theta**2 * K + (theta - theta.sin())/theta**3 * K@K
        return torch.where(theta>torch.finfo(theta.dtype).eps, Jr, I)


class SE3Type(LieType):
    def __init__(self):
        super().__init__(7, 7, 6)

    def Log(self, X):
        X = X.tensor() if hasattr(X, 'ltype') else X
        x = SE3_Log.apply(X)
        return LieTensor(x, ltype=se3_type)

    def Act(self, X, p):
        assert not self.on_manifold and isinstance(p, torch.Tensor)
        assert p.shape[-1]==3 or p.shape[-1]==4, "Invalid Tensor Dimension"
        X = X.tensor() if hasattr(X, 'ltype') else X
        input, out_shape = broadcast_inputs(X, p)
        if p.shape[-1]==3:
            out = SE3_Act.apply(*input)
        else:
            out = SE3_Act4.apply(*input)
        dim = -1 if out.nelement() != 0 else X.shape[-1]
        return out.view(out_shape + (dim,))

    def Mul(self, X, Y):
        # Transform on transform
        X = X.tensor() if hasattr(X, 'ltype') else X
        if not self.on_manifold and isinstance(Y, LieTensor) and not Y.ltype.on_manifold:
            Y = Y.tensor() if hasattr(Y, 'ltype') else Y
            input, out_shape = broadcast_inputs(X, Y)
            out = SE3_Mul.apply(*input)
            dim = -1 if out.nelement() != 0 else X.shape[-1]
            out = out.view(out_shape + (dim,))
            return LieTensor(out, ltype=SE3_type)
        # Transform on points
        if not self.on_manifold and isinstance(Y, torch.Tensor):
            return self.Act(X, Y)
        # (scalar or tensor) * manifold
        if self.on_manifold:
            return LieTensor(torch.mul(X, Y), ltype=SE3_type)
        raise NotImplementedError('Invalid __mul__ operation')

    def Inv(self, X):
        X = X.tensor() if hasattr(X, 'ltype') else X
        out = SE3_Inv.apply(X)
        return LieTensor(out, ltype=SE3_type)

    def rotation(self, input):
        return LieTensor(input.tensor()[..., 3:7], ltype=SO3_type)

    def translation(self, input):
        return input.tensor()[..., 0:3]

    def Adj(self, X, a):
        X = X.tensor() if hasattr(X, 'ltype') else X
        a = a.tensor() if hasattr(a, 'ltype') else a
        input, out_shape = broadcast_inputs(X, a)
        out = SE3_AdjXa.apply(*input)
        dim = -1 if out.nelement() != 0 else X.shape[-1]
        out = out.view(out_shape + (dim,))
        return LieTensor(out, ltype=se3_type)

    def AdjT(self, X, a):
        X = X.tensor() if hasattr(X, 'ltype') else X
        a = a.tensor() if hasattr(a, 'ltype') else a
        input, out_shape = broadcast_inputs(X, a)
        out = SE3_AdjTXa.apply(*input)
        dim = -1 if out.nelement() != 0 else X.shape[-1]
        out = out.view(out_shape + (dim,))
        return LieTensor(out, ltype=se3_type)

    def Jinvp(self, X, a):
        X = X.tensor() if hasattr(X, 'ltype') else X
        a = a.tensor() if hasattr(a, 'ltype') else a
        (X, a), out_shape = broadcast_inputs(X, a)
        out = (se3_Jl_inv(SE3_Log.apply(X)) @ a.unsqueeze(-1)).squeeze(-1)
        dim = -1 if out.nelement() != 0 else X.shape[-1]
        out = out.view(out_shape + (dim,))
        return LieTensor(out, ltype=se3_type)

    @classmethod
    def identity(cls, *size, **kwargs):
        data = torch.tensor([0., 0., 0., 0., 0., 0., 1.], **kwargs)
        return LieTensor(data.repeat(size+(1,)), ltype=SE3_type)

    def randn(self, *size, sigma=1.0, requires_grad=False, **kwargs):
        data = se3_type.Exp(se3_type.randn(*size, sigma=sigma, **kwargs)).detach()
        return LieTensor(data, ltype=SE3_type).requires_grad_(requires_grad)

    @classmethod
    def add_(cls, input, other):
        return input.copy_(LieTensor(other[..., :6], ltype=se3_type).Exp() * input)


class se3Type(LieType):
    def __init__(self):
        super().__init__(6, 7, 6)

    def Exp(self, x):
        x = x.tensor() if hasattr(x, 'ltype') else x
        X = se3_Exp.apply(x)
        return LieTensor(X, ltype=SE3_type)

    def Mul(self, X, Y):
        X = X.tensor() if hasattr(X, 'ltype') else X
        # (scalar or tensor) * manifold
        if self.on_manifold:
            return LieTensor(torch.mul(X, Y), ltype=se3_type)
        raise NotImplementedError('Invalid __mul__ operation')

    def rotation(self, input):
        return input.Exp().rotation()

    def translation(self, input):
        return input.Exp().translation()

    @classmethod
    def identity(cls, *size, **kwargs):
        return SE3_type.Log(SE3_type.identity(*size, **kwargs))

    def randn(self, *size, sigma=1.0, requires_grad=False, **kwargs):
        #  convert different types of inputs to SE3 sigma
        if not isinstance(sigma, collections.abc.Iterable):
            sigma = _quadruple(sigma)
        elif len(sigma)==2:
            rotation_sigma = _single(sigma[-1])
            translation_sigma = _triple(sigma[0])
            sigma = translation_sigma + rotation_sigma
        else:
            assert len(sigma)==4, 'Only accepts a tuple of sigma in size 1, 2, or 4.'
        size = self.to_tuple(size)
        rotation = so3_type.randn(*size, sigma=sigma[-1], **kwargs).detach().tensor()
        sigma = torch.tensor([sigma[0], sigma[1], sigma[2]], **kwargs)
        translation = sigma * torch.randn(*(size + torch.Size([3])), **kwargs)
        data = torch.cat([translation, rotation], dim=-1)
        return LieTensor(data, ltype=se3_type).requires_grad_(requires_grad)


class Sim3Type(LieType):
    def __init__(self):
        super().__init__(8, 8, 7)

    def Log(self, X):
        X = X.tensor() if hasattr(X, 'ltype') else X
        x = Sim3_Log.apply(X)
        return LieTensor(x, ltype=sim3_type)

    def Act(self, X, p):
        assert not self.on_manifold and isinstance(p, torch.Tensor)
        assert p.shape[-1]==3 or p.shape[-1]==4, "Invalid Tensor Dimension"
        X = X.tensor() if hasattr(X, 'ltype') else X
        input, out_shape = broadcast_inputs(X, p)
        if p.shape[-1]==3:
            out = Sim3_Act.apply(*input)
        else:
            out = Sim3_Act4.apply(*input)
        dim = -1 if out.nelement() != 0 else X.shape[-1]
        return out.view(out_shape + (dim,))

    def Mul(self, X, Y):
        # Transform on transform
        X = X.tensor() if hasattr(X, 'ltype') else X
        if not self.on_manifold and isinstance(Y, LieTensor) and not Y.ltype.on_manifold:
            Y = Y.tensor() if hasattr(Y, 'ltype') else Y
            input, out_shape = broadcast_inputs(X, Y)
            out = Sim3_Mul.apply(*input)
            dim = -1 if out.nelement() != 0 else X.shape[-1]
            out = out.view(out_shape + (dim,))
            return LieTensor(out, ltype=Sim3_type)
        # Transform on points
        if not self.on_manifold and isinstance(Y, torch.Tensor):
            return self.Act(X, Y)
        # (scalar or tensor) * manifold
        if self.on_manifold:
            return LieTensor(torch.mul(X, Y), ltype=Sim3_type)
        raise NotImplementedError('Invalid __mul__ operation')

    def Inv(self, X):
        X = X.tensor() if hasattr(X, 'ltype') else X
        out = Sim3_Inv.apply(X)
        return LieTensor(out, ltype=Sim3_type)

    def Adj(self, X, a):
        X = X.tensor() if hasattr(X, 'ltype') else X
        a = a.tensor() if hasattr(a, 'ltype') else a
        input, out_shape = broadcast_inputs(X, a)
        out = Sim3_AdjXa.apply(*input)
        dim = -1 if out.nelement() != 0 else X.shape[-1]
        out = out.view(out_shape + (dim,))
        return LieTensor(out, ltype=sim3_type)

    def AdjT(self, X, a):
        X = X.tensor() if hasattr(X, 'ltype') else X
        a = a.tensor() if hasattr(a, 'ltype') else a
        input, out_shape = broadcast_inputs(X, a)
        out = Sim3_AdjTXa.apply(*input)
        dim = -1 if out.nelement() != 0 else X.shape[-1]
        out = out.view(out_shape + (dim,))
        return LieTensor(out, ltype=sim3_type)

    def Jinvp(self, X, a):
        X = X.tensor() if hasattr(X, 'ltype') else X
        a = a.tensor() if hasattr(a, 'ltype') else a
        (X, a), out_shape = broadcast_inputs(X, a)
        out = (sim3_Jl_inv(Sim3_Log.apply(X)) @ a.unsqueeze(-1)).squeeze(-1)
        dim = -1 if out.nelement() != 0 else X.shape[-1]
        out = out.view(out_shape + (dim,))
        return LieTensor(out, ltype=sim3_type)

    def rotation(self, input):
        return LieTensor(input.tensor()[..., 3:7], ltype=SO3_type)

    def translation(self, input):
        return input.tensor()[..., 0:3]

    def scale(self, input):
        return input.tensor()[..., 7:8]

    @classmethod
    def identity(cls, *size, **kwargs):
        data = torch.tensor([0., 0., 0., 0., 0., 0., 1., 1.], **kwargs)
        return LieTensor(data.repeat(size+(1,)), ltype=Sim3_type)

    def randn(self, *size, sigma=1.0, requires_grad=False, **kwargs):
        data = sim3_type.Exp(sim3_type.randn(*size, sigma=sigma, **kwargs)).detach()
        return LieTensor(data, ltype=Sim3_type).requires_grad_(requires_grad)

    @classmethod
    def add_(cls, input, other):
        return input.copy_(LieTensor(other[..., :7], ltype=sim3_type).Exp() * input)


class sim3Type(LieType):
    def __init__(self):
        super().__init__(7, 8, 7)

    def Exp(self, x):
        x = x.tensor() if hasattr(x, 'ltype') else x
        X = sim3_Exp.apply(x)
        return LieTensor(X, ltype=Sim3_type)

    def Mul(self, X, Y):
        X = X.tensor() if hasattr(X, 'ltype') else X
        # (scalar or tensor) * manifold
        if self.on_manifold:
            return LieTensor(torch.mul(X, Y), ltype=sim3_type)
        raise NotImplementedError('Invalid __mul__ operation')

    def rotation(self, input):
        return input.Exp().rotation()

    def translation(self, input):
        return input.Exp().translation()

    def scale(self, input):
        return input.Exp().scale()

    @classmethod
    def identity(cls, *size, **kwargs):
        return Sim3_type.Log(Sim3_type.identity(*size, **kwargs))

    def randn(self, *size, sigma=1.0, requires_grad=False, **kwargs):
        if not isinstance(sigma, collections.abc.Iterable):
            sigma = _ntuple(5, "_penta")(sigma)
        elif len(sigma)==3:
            rotation_sigma = _single(sigma[-2])
            scale_sigma = _single(sigma[-1])
            translation_sigma = _triple(sigma[0])
            sigma = translation_sigma+rotation_sigma+scale_sigma
        else:
            assert len(sigma)==5, 'Only accepts a tuple of sigma in size 1, 3, or 5.'
        size = self.to_tuple(size)
        rotation = so3_type.randn(*size, sigma=sigma[-2], **kwargs).detach().tensor()
        scale = sigma[-1] * torch.randn(*(size + torch.Size([1])), **kwargs)
        sigma = torch.tensor([sigma[0], sigma[1], sigma[2]], **kwargs)
        translation = sigma * torch.randn(*(size + torch.Size([3])), **kwargs)
        data = torch.cat([translation, rotation, scale], dim=-1)
        return LieTensor(data, ltype=sim3_type).requires_grad_(requires_grad)


class RxSO3Type(LieType):
    def __init__(self):
        super().__init__(5, 5, 4)

    def Log(self, X):
        X = X.tensor() if hasattr(X, 'ltype') else X
        x = RxSO3_Log.apply(X)
        return LieTensor(x, ltype=rxso3_type)

    def Act(self, X, p):
        assert not self.on_manifold and isinstance(p, torch.Tensor)
        assert p.shape[-1]==3 or p.shape[-1]==4, "Invalid Tensor Dimension"
        X = X.tensor() if hasattr(X, 'ltype') else X
        input, out_shape = broadcast_inputs(X, p)
        if p.shape[-1]==3:
            out = RxSO3_Act.apply(*input)
        else:
            out = RxSO3_Act4.apply(*input)
        dim = -1 if out.nelement() != 0 else X.shape[-1]
        return out.view(out_shape + (dim,))

    def Mul(self, X, Y):
        # Transform on transform
        X = X.tensor() if hasattr(X, 'ltype') else X
        if not self.on_manifold and isinstance(Y, LieTensor) and not Y.ltype.on_manifold:
            Y = Y.tensor() if hasattr(Y, 'ltype') else Y
            input, out_shape = broadcast_inputs(X, Y)
            out = RxSO3_Mul.apply(*input)
            dim = -1 if out.nelement() != 0 else X.shape[-1]
            out = out.view(out_shape + (dim,))
            return LieTensor(out, ltype=RxSO3_type)
        # Transform on points
        if not self.on_manifold and isinstance(Y, torch.Tensor):
            return self.Act(X, Y)
        # (scalar or tensor) * manifold
        if self.on_manifold:
            return LieTensor(torch.mul(X, Y), ltype=RxSO3_type)
        raise NotImplementedError('Invalid __mul__ operation')

    def Inv(self, X):
        X = X.tensor() if hasattr(X, 'ltype') else X
        out = RxSO3_Inv.apply(X)
        return LieTensor(out, ltype=RxSO3_type)

    def Adj(self, X, a):
        X = X.tensor() if hasattr(X, 'ltype') else X
        a = a.tensor() if hasattr(a, 'ltype') else a
        input, out_shape = broadcast_inputs(X, a)
        out = RxSO3_AdjXa.apply(*input)
        dim = -1 if out.nelement() != 0 else X.shape[-1]
        out = out.view(out_shape + (dim,))
        return LieTensor(out, ltype=rxso3_type)

    def AdjT(self, X, a):
        X = X.tensor() if hasattr(X, 'ltype') else X
        a = a.tensor() if hasattr(a, 'ltype') else a
        input, out_shape = broadcast_inputs(X, a)
        out = RxSO3_AdjTXa.apply(*input)
        dim = -1 if out.nelement() != 0 else X.shape[-1]
        out = out.view(out_shape + (dim,))
        return LieTensor(out, ltype=rxso3_type)

    def Jinvp(self, X, a):
        X = X.tensor() if hasattr(X, 'ltype') else X
        a = a.tensor() if hasattr(a, 'ltype') else a
        (X, a), out_shape = broadcast_inputs(X, a)
        out = (rxso3_Jl_inv(RxSO3_Log.apply(X)) @ a.unsqueeze(-1)).squeeze(-1)
        dim = -1 if out.nelement() != 0 else X.shape[-1]
        out = out.view(out_shape + (dim,))
        return LieTensor(out, ltype=rxso3_type)

    def rotation(self, input):
        return LieTensor(input.tensor()[..., 0:4], ltype=SO3_type)

    def scale(self, input):
        return input.tensor()[..., 4:5]

    @classmethod
    def identity(cls, *size, **kwargs):
        data = torch.tensor([0., 0., 0., 1., 1.], **kwargs)
        return LieTensor(data.repeat(size+(1,)), ltype=RxSO3_type)

    def randn(self, *size, sigma=1.0, requires_grad=False, **kwargs):
        data = rxso3_type.Exp(rxso3_type.randn(*size, sigma=sigma, **kwargs)).detach()
        return LieTensor(data, ltype=RxSO3_type).requires_grad_(requires_grad)

    @classmethod
    def add_(cls, input, other):
        return input.copy_(LieTensor(other[..., :4], ltype=rxso3_type).Exp() * input)


class rxso3Type(LieType):
    def __init__(self):
        super().__init__(4, 5, 4)

    def Exp(self, x):
        x = x.tensor() if hasattr(x, 'ltype') else x
        X = rxso3_Exp.apply(x)
        return LieTensor(X, ltype=RxSO3_type)

    def Mul(self, X, Y):
        X = X.tensor() if hasattr(X, 'ltype') else X
        # (scalar or tensor) * manifold
        if self.on_manifold:
            return LieTensor(torch.mul(X, Y), ltype=rxso3_type)
        raise NotImplementedError('Invalid __mul__ operation')

    def rotation(self, input):
        return input.Exp().rotation()

    def scale(self, input):
        return input.Exp().scale()

    @classmethod
    def identity(cls, *size, **kwargs):
        return RxSO3_type.Log(RxSO3_type.identity(*size, **kwargs))

    def randn(self, *size, sigma=1.0, requires_grad=False, **kwargs):
        if not isinstance(sigma, collections.abc.Iterable):
            sigma = _pair(sigma)
        else:
            assert len(sigma)==2, 'Only accepts a tuple of sigma in size 1 or 2.'
        size = self.to_tuple(size)
        rotation = so3_type.randn(*size, sigma=sigma[0], **kwargs).tensor()
        scale = sigma[1] * torch.randn(*(size + torch.Size([1])), **kwargs)
        data = torch.cat([rotation, scale], dim=-1)
        return LieTensor(data, ltype=rxso3_type).requires_grad_(requires_grad)


SO3_type, so3_type = SO3Type(), so3Type()
SE3_type, se3_type = SE3Type(), se3Type()
Sim3_type, sim3_type = Sim3Type(), sim3Type()
RxSO3_type, rxso3_type = RxSO3Type(), rxso3Type()


class LieTensor(torch.Tensor):
    r""" A sub-class of :obj:`torch.Tensor` to represent Lie Algebra and Lie Group.

    Args:
        data (:obj:`Tensor`, or :obj:`list`, or ':obj:`int`...'): A
            :obj:`Tensor` object, or constructing a :obj:`Tensor`
            object from :obj:`list`, which defines tensor data, or from
            ':obj:`int`...', which defines tensor shape.

            The shape of :obj:`Tensor` object should be compatible
            with Lie Type :obj:`ltype`, otherwise error will be raised.

        ltype (:obj:`ltype`): Lie Type, either **Lie Group** or **Lie Algebra** is listed below:

    Returns:
        LieTensor corresponding to Lie Type :obj:`ltype`.

    .. list-table:: List of :obj:`ltype` for **Lie Group**
        :widths: 25 25 30 30
        :header-rows: 1

        * - Representation
          - :obj:`ltype`
          - :obj:`shape`
          - Alias Class
        * - Rotation
          - :obj:`SO3_type`
          - :obj:`(*, 4)`
          - :meth:`SO3`
        * - Translation + Rotation
          - :obj:`SE3_type`
          - :obj:`(*, 7)`
          - :meth:`SE3`
        * - Translation + Rotation + Scale
          - :obj:`Sim3_type`
          - :obj:`(*, 8)`
          - :meth:`Sim3`
        * - Rotation + Scale
          - :obj:`RxSO3_type`
          - :obj:`(*, 5)`
          - :meth:`RxSO3`

    .. list-table:: List of :obj:`ltype` for **Lie Algebra**
        :widths: 25 25 30 30
        :header-rows: 1

        * - Representation
          - :obj:`ltype`
          - :obj:`shape`
          - Alias Class
        * - Rotation
          - :obj:`so3_type`
          - :obj:`(*, 3)`
          - :meth:`so3`
        * - Translation + Rotation
          - :obj:`se3_type`
          - :obj:`(*, 6)`
          - :meth:`se3`
        * - Translation + Rotation + Scale
          - :obj:`sim3_type`
          - :obj:`(*, 7)`
          - :meth:`sim3`
        * - Rotation + Scale
          - :obj:`rxso3_type`
          - :obj:`(*, 4)`
          - :meth:`rxso3`

    Note:
        Two attributes :obj:`shape` and :obj:`lshape` are available for LieTensor.
        The only differece is the :obj:`lshape` hides the last dimension of :obj:`shape`,
        since :obj:`lshape` takes the data in the last dimension as a single :obj:`ltype` item.

        See LieTensor method :meth:`lview` for more details.

    Examples:
        >>> import torch
        >>> import pypose as pp
        >>> data = torch.randn(3, 3, requires_grad=True, device='cuda:0')
        >>> pp.LieTensor(data, ltype=pp.so3_type)
        so3Type LieTensor:
        tensor([[ 0.9520,  0.4517,  0.5834],
                [-0.8106,  0.8197,  0.7077],
                [-0.5743,  0.8182, -1.2104]], device='cuda:0', grad_fn=<AliasBackward0>)

        Alias class for specific LieTensor is recommended:

        >>> pp.so3(data)
        so3Type LieTensor:
        tensor([[ 0.9520,  0.4517,  0.5834],
                [-0.8106,  0.8197,  0.7077],
                [-0.5743,  0.8182, -1.2104]], device='cuda:0', grad_fn=<AliasBackward0>)

        See more alias classes at `Table 1 for Lie Group <#id1>`_  and `Table 2 for Lie Algebra <#id2>`_.

        Other constructors:

            - From list.

            >>> pp.so3([0, 0, 0])
            so3Type LieTensor:
            tensor([0., 0., 0.])

            - From ints.

            >>> pp.so3(2, 3)
            so3Type LieTensor:
            tensor([[0., 0., 0.],
                    [0., 0., 0.]])

    Note:
        Alias class for LieTensor is recommended.
        For example, the following usage is equivalent:

        - :obj:`pp.LieTensor(tensor, ltype=pp.so3_type)`

        - :obj:`pp.so3(tensor)` (This is preferred).

    Note:
        All attributes from Tensor are available for LieTensor, e.g., :obj:`dtype`,
        :obj:`device`, and :obj:`requires_grad`. See more details at
        `tensor attributes <https://pytorch.org/docs/stable/tensor_attributes.html>`_.

        Example:

            >>> data = torch.randn(1, 3, dtype=torch.float64, device="cuda", requires_grad=True)
            >>> pp.so3(data) # All Tensor attributes are available for LieTensor
            so3Type LieTensor:
            tensor([[-1.5948,  0.3113, -0.9807]], device='cuda:0', dtype=torch.float64,
                grad_fn=<AliasBackward0>)

    Note:
        In most of the cases, Lie Group is expected to be used, therefore we only provide
        `converting functions <https://pypose.org/docs/main/convert/>`_ between Lie Groups
        and other data structures, e.g., transformation matrix, Euler angle, etc. The users
        can convert data between Lie Group and Lie algebra with :obj:`Exp` and :obj:`Log`.
    """
    def __init__(self, *data, ltype:LieType):
        assert self.shape[-1:] == ltype.dimension, 'The last dimension of a LieTensor has to be ' \
            'corresponding to their LieType. More details go to {}. If this error happens in an ' \
            'optimization process, where LieType is not a necessary structure, we suggest to '    \
            'call .tensor() to convert a LieTensor to Tensor before passing it to an optimizer. ' \
            'If this still happens, create an issue on GitHub please.'.format(
            'https://pypose.org/docs/main/generated/pypose.LieTensor')
        self.ltype = ltype

    @staticmethod
    def __new__(cls, *data, ltype):
        tensor = data[0] if isinstance(data[0], torch.Tensor) else torch.Tensor(*data)
        return torch.Tensor.as_subclass(tensor, LieTensor)

    def __repr__(self):
        if hasattr(self, 'ltype'):
            return self.ltype.__class__.__name__ + \
                   ' %s:\n'%(self.__class__.__name__) + super().__repr__()
        else:
            return super().__repr__()

    def new_empty(self, shape):
        return torch.Tensor.as_subclass(torch.empty(shape), LieTensor)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs={}):
        ltypes = (torch.Tensor if t is LieTensor or Parameter else t for t in types)
        data = torch.Tensor.__torch_function__(func, ltypes, args, kwargs)
        if data is not None and func.__name__ in HANDLED_FUNCTIONS:
            args, spec = tree_flatten(args)
            ltype = [arg.ltype for arg in args if isinstance(arg, LieTensor)][0]
            def warp(t):
                if isinstance(t, torch.Tensor) and not isinstance(t, cls):
                    lt = torch.Tensor.as_subclass(t, LieTensor)
                    lt.ltype = ltype
                    if lt.shape[-1:] != lt.ltype.dimension:
                        link = 'https://pypose.org/docs/main/generated/pypose.LieTensor'
                        warnings.warn('Tensor Shape Invalid by calling {}, ' \
                            'go to {}'.format(func, link))
                    return lt
                return t
            return tree_map(warp, data)
        return data

    @property
    def lshape(self) -> torch.Size:
        r'''
        LieTensor Shape (shape of torch.Tensor by ignoring the last dimension)

        Returns:
            torch.Size

        Note:
            - The only difference from :obj:`shape` is the last dimension is hidden,
              since :obj:`lshape` takes the last dimension as a single :obj:`ltype` item.

            - The last dimension can also be accessed via :obj:`LieTensor.ltype.dimension`.

        Examples:
            >>> x = pp.randn_SE3(2)
            >>> x.lshape
            torch.Size([2])
            >>> x.shape
            torch.Size([2, 7])
            >>> x.ltype.dimension
            torch.Size([7])
        '''
        return self.shape[:-1]

    def lview(self, *shape):
        r'''
        Returns a new LieTensor with the same data as the self tensor but of a different
        :obj:`lshape`.

        Args:
            shape (torch.Size or int...): the desired size

        Returns:
            A new lieGroup tensor sharing with the same data as the self tensor but of a
            different shape.

        Note:
            The only difference from :meth:`view` is the last dimension is hidden.

            See `Tensor.view <https://tinyurl.com/mrds8nmd>`_
            for its usage.

        Examples:
            >>> x = pp.randn_so3(2,2)
            >>> x.shape
            torch.Size([2, 2, 3])
            >>> x.lview(-1).lshape
            torch.Size([4])
        '''
        return self.view(*shape+self.ltype.dimension)

    def Exp(self):
        r'''
        See :meth:`pypose.Exp`
        '''
        return self.ltype.Exp(self)

    def Log(self):
        r'''
        See :meth:`pypose.Log`
        '''
        return self.ltype.Log(self)

    def Inv(self):
        r'''
        See :meth:`pypose.Inv`
        '''
        return self.ltype.Inv(self)

    def Act(self, p):
        r'''
        See :meth:`pypose.Act`
        '''
        return self.ltype.Act(self, p)

    def add(self, other, alpha=1):
        r'''
        See :meth:`pypose.add`
        '''
        return self.clone().add_(other = alpha * other)

    def add_(self, other, alpha=1):
        r'''
        See :meth:`pypose.add_`
        '''
        return self.ltype.add_(self, other = alpha * other)

    def __add__(self, other):
        return self.add(other=other)

    def __mul__(self, other):
        r'''
        See :meth:`pypose.mul`
        '''
        return self.ltype.Mul(self, other)

    def __matmul__(self, other):
        r'''
        See :meth:`pypose.matmul`
        '''
        if isinstance(other, LieTensor):
            return self.ltype.Mul(self, other)
        else: # Same with: self.ltype.matrix(self) @ other
            return self.Act(other)

    def Retr(self, a):
        r'''
        See :meth:`pypose.Retr`
        '''
        return self.ltype.Retr(self, a)

    def Adj(self, a):
        r'''
        See :meth:`pypose.Adj`
        '''
        return self.ltype.Adj(self, a)

    def AdjT(self, a):
        r'''
        See :meth:`pypose.AdjT`
        '''
        return self.ltype.AdjT(self, a)

    def Jinvp(self, p):
        r'''
        See :meth:`pypose.Jinvp`
        '''
        return self.ltype.Jinvp(self, p)

    def Jr(self):
        r'''
        See :meth:`pypose.Jr`
        '''
        return self.ltype.Jr(self)

    def tensor(self) -> torch.Tensor:
        r'''
        See :meth:`pypose.tensor`
        '''
        return torch.Tensor.as_subclass(self, torch.Tensor)

    def matrix(self) -> torch.Tensor:
        r'''
        See :meth:`pypose.matrix`
        '''
        return self.ltype.matrix(self)

    def translation(self) -> torch.Tensor:
        r'''
        See :meth:`pypose.translation`
        '''
        return self.ltype.translation(self)

    def rotation(self):
        r'''
        See :meth:`pypose.rotation`
        '''
        return self.ltype.rotation(self)

    def scale(self) -> torch.Tensor:
        r'''
        See :meth:`pypose.scale`
        '''
        return self.ltype.scale(self)

    def euler(self, eps=2e-4) -> torch.Tensor:
        r'''
        See :meth:`pypose.euler`
        '''
        data = self.rotation().tensor()
        x, y = data[..., 0], data[..., 1]
        z, w = data[..., 2], data[..., 3]
        xx, yy, zz, ww = x*x, y*y, z*z, w*w

        t0 = 2 * (w * x + y * z)
        t1 = (ww + zz) - (xx + yy)
        t2 = 2 * (w * y - z * x) / (xx + yy + zz + ww)
        t3 = 2 * (w * z + x * y)
        t4 = (ww + xx) - (yy + zz)

        # sigularity when pitch angle ~ +/-pi/2
        roll1 = torch.atan2(t0, t1)
        roll2 = torch.zeros_like(t0)
        flag = t2.abs() < 1. - eps
        yaw1 = torch.atan2(t3, t4)
        yaw2 = -2 * pm(t2) * torch.atan2(x, w)
        
        roll = torch.where(flag, roll1, roll2)
        pitch = torch.asin(t2.clamp(-1, 1))
        yaw = torch.where(flag, yaw1, yaw2)

        return torch.stack([roll, pitch, yaw], dim=-1)

    def identity_(self):
        r'''
        Inplace set the LieTensor to identity.

        Return:
            LieTensor: the :obj:`self` LieTensor

        Note:
            The translation part, if there is, is set to zeros, while the
            rotation part is set to identity quaternion.

        Example:
            >>> x = pp.randn_SO3(2)
            >>> x
            SO3Type LieTensor:
            tensor([[-0.0724,  0.1970,  0.0022,  0.9777],
                    [ 0.3492,  0.4998, -0.5310,  0.5885]])
            >>> x.identity_()
            SO3Type LieTensor:
            tensor([[0., 0., 0., 1.],
                    [0., 0., 0., 1.]])
        '''
        return self.ltype.identity_(self)

    def cumops(self, dim, ops):
        r"""
        See :func:`pypose.cumops`
        """
        return self.ltype.cumops(self, dim, ops)

    def cummul(self, dim):
        r"""
        See :func:`pypose.cummul`
        """
        return self.ltype.cummul(self, dim)

    def cumprod(self, dim, left = True):
        r"""
        See :func:`pypose.cumprod`
        """
        return self.ltype.cumprod(self, dim, left)

    def cumops_(self, dim, ops):
        r"""
        Inplace version of :func:`pypose.cumops`
        """
        return self.ltype.cumops_(self, dim, ops)

    def cummul_(self, dim):
        r"""
        Inplace version of :func:`pypose.cummul`
        """
        return self.ltype.cummul_(self, dim)

    def cumprod_(self, dim):
        r"""
        Inplace version of :func:`pypose.cumprod`
        """
        return self.ltype.cumprod_(self, dim)


class Parameter(LieTensor, nn.Parameter):
    r'''
    A kind of LieTensor that is to be considered a module parameter.

    Parameters are of :meth:`LieTensor` and :meth:`torch.nn.Parameter`,
    that have a very special property when used with Modules: when
    they are assigned as Module attributes they are automatically
    added to the list of its parameters, and will appear, e.g., in
    :meth:`parameters()` iterator.

    Args:
        data (LieTensor): parameter LieTensor.
        requires_grad (bool, optional): if the parameter requires
            gradient. Default: ``True``

    Examples:
        >>> import torch, pypose as pp
        >>> x = pp.Parameter(pp.randn_SO3(2))
        >>> x.Log().sum().backward()
        >>> x.grad
        tensor([[0.8590, 1.4069, 0.6261, 0.0000],
                [1.2869, 1.0748, 0.5385, 0.0000]])
    '''
    def __init__(self, data, **kwargs):
        self.ltype = data.ltype

    def __new__(cls, data=None, requires_grad=True):
        if data is None:
            data = torch.tensor([])
        return LieTensor._make_subclass(cls, data, requires_grad)

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        else:
            result = type(self)(self.clone(memory_format=torch.preserve_format))
            memo[id(self)] = result
            return result
