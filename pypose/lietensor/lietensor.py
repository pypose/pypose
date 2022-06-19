
import math, numbers
import torch, warnings
from torch import nn, linalg
from torch.utils._pytree import tree_map, tree_flatten
from .backends import exp, log, inv, mul, adj
from .backends import adjT, jinvp, act3, act4, toMatrix
from .basics import vec2skew, cumops, cummul, cumprod
from .basics import cumops_, cummul_, cumprod_


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
    def __init__(self, lid, dimension, embedding, manifold):
        self._lid       = lid                     # LieType ID
        self._dimension = torch.Size([dimension]) # Data dimension
        self._embedding = torch.Size([embedding]) # Embedding dimension
        self._manifold  = torch.Size([manifold])  # Manifold dimension

    @property
    def lid(self):
        return self._lid

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
            return LieTensor(-x, ltype=x.ltype)
        out = self.__op__(self.lid, inv, x)
        return LieTensor(out, ltype=x.ltype)

    def Act(self, x, p):
        """ action on a points tensor(*, 3[4]) (homogeneous)"""
        assert not self.on_manifold and isinstance(p, torch.Tensor)
        assert p.shape[-1]==3 or p.shape[-1]==4, "Invalid Tensor Dimension"
        act = act3 if p.shape[-1]==3 else act4
        return self.__op__(self.lid, act, x, p)

    def Mul(self, x, y):
        # Transform on transform
        if not self.on_manifold and isinstance(y, LieTensor) and not y.ltype.on_manifold:
            out = self.__op__(self.lid, mul, x, y)
            return LieTensor(out, ltype=x.ltype)
        # Transform on points
        if not self.on_manifold and isinstance(y, torch.Tensor):
            return self.Act(x, y)
        # (scalar or tensor) * manifold
        if self.on_manifold:
            return LieTensor(torch.mul(x, y), ltype=x.ltype)
        raise NotImplementedError('Invalid __mul__ operation')

    def Retr(self, X, a):
        if self.on_manifold:
            raise AttributeError("Has no Retr attribute")
        return a.Exp() * X

    def Adj(self, X, a):
        ''' X * Exp(a) = Exp(Adj) * X '''
        if self.on_manifold:
            raise AttributeError("Has no Adj attribute")
        assert not X.ltype.on_manifold and a.ltype.on_manifold
        assert X.ltype.lid == a.ltype.lid
        out = self.__op__(self.lid, adj, X, a)
        return LieTensor(out, ltype=a.ltype)

    def AdjT(self, X, a):
        ''' Exp(a) * X = X * Exp(AdjT) '''
        if self.on_manifold:
            raise AttributeError("Has no AdjT attribute")
        assert not X.ltype.on_manifold and a.ltype.on_manifold, "ltype Invalid"
        assert X.ltype.lid == a.ltype.lid, "ltype Invalid"
        out = self.__op__(self.lid, adjT, X, a)
        return LieTensor(out, ltype=a.ltype)

    def Jinvp(self, X, p):
        if self.on_manifold:
            raise AttributeError("ltype has no Jinvp attribute")
        assert isinstance(p, LieTensor) and p.ltype.on_manifold, "Args p has to be Lie Algebra"
        out = self.__op__(self.lid, jinvp, X, p)
        return LieTensor(out, ltype=p.ltype)

    def matrix(self, lietensor):
        """ To 4x4 matrix """
        X = lietensor.Exp() if self.on_manifold else lietensor
        I = torch.eye(4, dtype=X.dtype, device=X.device)
        I = I.view([1] * (X.dim() - 1) + [4, 4])
        return X.unsqueeze(-2).Act(I).transpose(-1,-2)

    def rotation(self, lietensor):
        raise NotImplementedError("Rotation is not implemented for the instance.")

    def translation(self, lietensor):
        warnings.warn("Instance has no translation. Zero vector(s) is returned.")
        return torch.zeros(lietensor.lshape + (3,))

    def scale(self, lietensor):
        warnings.warn("Instance has no scale. Scalar one(s) is returned.")
        return torch.ones(lietensor.lshape + (1,))

    @classmethod
    def identity(cls, *args, **kwargs):
        raise NotImplementedError("Instance has no identity.")

    @classmethod
    def identity_like(cls, *args, **kwargs):
        return cls.identity(*args, **kwargs)

    def randn_like(self, *args, sigma=1, **kwargs):
        return self.randn(*args, sigma=1, **kwargs)

    def randn(self, *args, sigma=1., **kwargs):
        scaled_sigma = 2.*sigma/math.sqrt(3)
        return scaled_sigma * torch.randn(*(tuple(args)+self.manifold), **kwargs)

    @classmethod
    def __op__(cls, lid, op, x, y=None):
        inputs, out_shape = cls.__broadcast_inputs(x, y)
        out = op.apply(lid, *inputs)
        dim = -1 if out.nelement() != 0 else x.shape[-1]
        return out.view(out_shape + (dim,))

    @classmethod
    def __broadcast_inputs(self, x, y):
        """ Automatic broadcasting of missing dimensions """
        if y is None:
            xs, xd = x.shape[:-1], x.shape[-1]
            return (x.reshape(-1, xd).contiguous(), ), x.shape[:-1]
        out_shape = torch.broadcast_shapes(x.shape[:-1], y.shape[:-1])
        shape = out_shape if out_shape != torch.Size([]) else (1,)
        x = x.expand(shape+(x.shape[-1],)).reshape(-1,x.shape[-1]).contiguous()
        y = y.expand(shape+(y.shape[-1],)).reshape(-1,y.shape[-1]).contiguous()
        return (x, y), tuple(out_shape)

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
        super().__init__(1, 4, 4, 3)

    def Log(self, X):
        x = self.__op__(self.lid, log, X)
        return LieTensor(x, ltype=so3_type)

    @classmethod
    def identity(cls, *size, **kwargs):
        data = torch.tensor([0., 0., 0., 1.], **kwargs)
        return LieTensor(data.repeat(size+(1,)), ltype=SO3_type)

    def randn(self, *size, sigma=1, requires_grad=False, **kwargs):
        data = so3_type.Exp(so3_type.randn(*size, sigma=sigma, **kwargs)).detach()
        return LieTensor(data, ltype=SO3_type).requires_grad_(requires_grad)

    @classmethod
    def add_(cls, input, other):
        return input.copy_(LieTensor(other[..., :3], ltype=so3_type).Exp() * inputs)

    def matrix(self, X):
        """ To 3x3 matrix """
        I = torch.eye(3, dtype=X.dtype, device=X.device)
        I = I.view([1] * (X.dim() - 1) + [3, 3])
        return X.unsqueeze(-2).Act(I).transpose(-1,-2)

    def rotation(self, X):
        return X

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
        super().__init__(1, 3, 4, 3)

    def Exp(self, x):
        X = self.__op__(self.lid, exp, x)
        return LieTensor(X, ltype=SO3_type)

    @classmethod
    def identity(cls, *size, **kwargs):
        return SO3_type.Log(SO3_type.identity(*size, **kwargs))

    def randn(self, *size, sigma=1, requires_grad=False, **kwargs):
        data = super().randn(*size, sigma=sigma, **kwargs).detach()
        return LieTensor(data, ltype=so3_type).requires_grad_(requires_grad)

    def matrix(self, lietensor):
        """ To 3x3 matrix """
        X = lietensor.Exp()
        I = torch.eye(3, dtype=X.dtype, device=X.device)
        I = I.view([1] * (X.dim() - 1) + [3, 3])
        return X.unsqueeze(-2).Act(I).transpose(-1,-2)

    def rotation(self, lietensor):
        return lietensor.Exp().rotation()

    def Jr(self, x):
        """
        Right jacobian of so(3)
        The code is taken from the Sophus codebase :
        https://github.com/XueLianjie/BA_schur/blob/3af9a94248d4a272c53cfc7acccea4d0208b77f7/thirdparty/Sophus/sophus/so3.hpp#L113
        """
        K = vec2skew(x)
        theta = torch.linalg.norm(x, dim=-1, keepdim=True).unsqueeze(-1)
        I = torch.eye(3, device=x.device, dtype=x.dtype).expand(x.lshape+(3, 3))
        Jr = I - (1-theta.cos())/theta**2 * K + (theta - theta.sin())/theta**3 * K@K
        return torch.where(theta>torch.finfo(theta.dtype).eps, Jr, I)


class SE3Type(LieType):
    def __init__(self):
        super().__init__(3, 7, 7, 6)

    def Log(self, X):
        x = self.__op__(self.lid, log, X)
        return LieTensor(x, ltype=se3_type)

    def rotation(self, X):
        return LieTensor(X.tensor()[..., 0:4], ltype=SO3_type)

    def translation(self, X):
        return X.tensor()[..., 4:6]

    @classmethod
    def identity(cls, *size, **kwargs):
        data = torch.tensor([0., 0., 0., 0., 0., 0., 1.], **kwargs)
        return LieTensor(data.repeat(size+(1,)), ltype=SE3_type)

    def randn(self, *size, sigma=1, requires_grad=False, **kwargs):
        data = se3_type.Exp(se3_type.randn(*size, sigma=sigma, **kwargs)).detach()
        return LieTensor(data, ltype=SE3_type).requires_grad_(requires_grad)

    @classmethod
    def add_(cls, input, other):
        return input.copy_(LieTensor(other[..., :6], ltype=se3_type).Exp() * input)


class se3Type(LieType):
    def __init__(self):
        super().__init__(3, 6, 7, 6)

    def Exp(self, x):
        X = self.__op__(self.lid, exp, x)
        return LieTensor(X, ltype=SE3_type)

    def rotation(self, lietensor):
        return lietensor.Exp().rotation()

    def translation(self, lietensor):
        return lietensor.Exp().translation()

    @classmethod
    def identity(cls, *size, **kwargs):
        return SE3_type.Log(SE3_type.identity(*size, **kwargs))

    def randn(self, *size, sigma=1, requires_grad=False, **kwargs):
        data = super().randn(*size, sigma=sigma, **kwargs).detach()
        return LieTensor(data, ltype=se3_type).requires_grad_(requires_grad)


class Sim3Type(LieType):
    def __init__(self):
        super().__init__(4, 8, 8, 7)

    def Log(self, X):
        x = self.__op__(self.lid, log, X)
        return LieTensor(x, ltype=sim3_type)

    def rotation(self, X):
        return LieTensor(X.tensor()[..., 0:4], ltype=SO3_type)

    def translation(self, X):
        return X.tensor()[..., 4:6]

    def scale(self, X):
        return X.tensor()[..., 6:7]

    @classmethod
    def identity(cls, *size, **kwargs):
        data = torch.tensor([0., 0., 0., 0., 0., 0., 1., 1.], **kwargs)
        return LieTensor(data.repeat(size+(1,)), ltype=Sim3_type)

    def randn(self, *size, sigma=1, requires_grad=False, **kwargs):
        data = sim3_type.Exp(sim3_type.randn(*size, sigma=sigma, **kwargs)).detach()
        return LieTensor(data, ltype=Sim3_type).requires_grad_(requires_grad)

    @classmethod
    def add_(cls, input, other):
        return input.copy_(LieTensor(other[..., :7], ltype=sim3_type).Exp() * input)


class sim3Type(LieType):
    def __init__(self):
        super().__init__(4, 7, 8, 7)

    def Exp(self, x):
        X = self.__op__(self.lid, exp, x)
        return LieTensor(X, ltype=Sim3_type)

    def rotation(self, lietensor):
        return lietensor.Exp().rotation()

    def translation(self, lietensor):
        return lietensor.Exp().translation()

    def scale(self, lietensor):
        return lietensor.Exp().scale()

    @classmethod
    def identity(cls, *size, **kwargs):
        return Sim3_type.Log(Sim3_type.identity(*size, **kwargs))

    def randn(self, *size, sigma=1, requires_grad=False, **kwargs):
        data = super().randn(*size, sigma=sigma, **kwargs).detach()
        return LieTensor(data, ltype=sim3_type).requires_grad_(requires_grad)


class RxSO3Type(LieType):
    def __init__(self):
        super().__init__(2, 5, 5, 4)

    def Log(self, X):
        x = self.__op__(self.lid, log, X)
        return LieTensor(x, ltype=rxso3_type)

    def rotation(self, X):
        return LieTensor(X.tensor()[..., 0:4], ltype=SO3_type)

    def scale(self, X):
        return X.tensor()[..., 4:5]

    @classmethod
    def identity(cls, *size, **kwargs):
        data = torch.tensor([0., 0., 0., 1., 1.], **kwargs)
        return LieTensor(data.repeat(size+(1,)), ltype=RxSO3_type)

    def randn(self, *size, sigma=1, requires_grad=False, **kwargs):
        data = rxso3_type.Exp(rxso3_type.randn(*size, sigma=sigma, **kwargs)).detach()
        return LieTensor(data, ltype=RxSO3_type).requires_grad_(requires_grad)

    @classmethod
    def add_(cls, input, other):
        return input.copy_(LieTensor(other[..., :4], ltype=rxso3_type).Exp() * input)


class rxso3Type(LieType):
    def __init__(self):
        super().__init__(2, 4, 5, 4)

    def Exp(self, x):
        X = self.__op__(self.lid, exp, x)
        return LieTensor(X, ltype=RxSO3_type)

    def rotation(self, lietensor):
        return lietensor.Exp().rotation()

    def scale(self, lietensor):
        return lietensor.Exp().scale()

    @classmethod
    def identity(cls, *size, **kwargs):
        return RxSO3_type.Log(RxSO3_type.identity(*size, **kwargs))

    def randn(self, *size, sigma=1, requires_grad=False, **kwargs):
        data = super().randn(*size, sigma=sigma, **kwargs).detach()
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
        In most of the cases, Lie Group should be used. Lie Algebra is used only
        when it needs to be optimized by back-propagation via gradients: in this
        case, LieTensor is taken as :meth:`pypose.Parameter` in a module, which follows
        PyTorch traditions.


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
    """
    def __init__(self, *data, ltype:LieType):
        assert self.shape[-1:] == ltype.dimension, 'Dimension Invalid. Go to{}'.format(
            'https://pypose.org/docs/generated/pypose.LieTensor/#pypose.LieTensor')
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
                        link = 'https://pypose.org/docs/generated/pypose.LieTensor/#pypose.LieTensor'
                        warnings.warn('Tensor Shape Invalid by calling {}, go to {}'.format(func, link))
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
        Returns a new LieTensor with the same data as the self tensor but of a different :obj:`lshape`.

        Args:
            shape (torch.Size or int...): the desired size

        Returns:
            A new lieGroup tensor sharing with the same data as the self tensor but of a different shape.

        Note:
            The only difference from :meth:`view` is the last dimension is hidden.

            See `Tensor.view <https://pytorch.org/docs/stable/generated/torch.Tensor.view.html?highlight=view#torch.Tensor.view>`_
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
        return self.ltype.Mul(self, other)

    def __matmul__(self, other):
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
        return self.ltype.cumops(self, other, dim, ops)

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
        return self.ltype.cumops_(self, other, dim, ops)

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
            gradient. Default: True

    Examples:
        >>> x = pp.Parameter(pp.randn_so3(2))
        >>> x.sum().backward() # Just test. There is no physical meaning
        >>> x.grad
        tensor([[1., 1., 1.],
                [1., 1., 1.]])
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
