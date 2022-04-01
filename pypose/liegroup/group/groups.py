import torch
from torch import nn
from .group_ops import exp, log, inv, mul, adj
from .group_ops import adjT, jinv, act3, act4, toMatrix
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
                     'select', 'select_scatter', 'index_put','index_put_']


class GroupType:
    '''Lie Group Type Base Class'''
    def __init__(self, groud, dimension, embedding, manifold):
        self._group     = groud     # Group ID
        self._dimension = torch.Size([dimension]) # Data dimension
        self._embedding = torch.Size([embedding]) # Embedding dimension
        self._manifold  = torch.Size([manifold])  # Manifold dimension

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
            return LieGroup(-x, gtype=x.gtype)
        out = self.__op__(self.group, inv, x)
        return LieGroup(out, gtype=x.gtype)

    def Act(self, x, p):
        """ action on a points tensor(*, 3[4]) (homogeneous)"""
        assert not self.on_manifold and isinstance(p, torch.Tensor)
        assert p.shape[-1]==3 or p.shape[-1]==4, "Invalid Tensor Dimension"
        act = act3 if p.shape[-1]==3 else act4
        return self.__op__(self.group, act, x, p)

    def Mul(self, x, y):
        # Transform on transform
        if not self.on_manifold and isinstance(y, LieGroup) and not y.gtype.on_manifold:
            out = self.__op__(self.group, mul, x, y)
            return LieGroup(out, gtype=x.gtype)
        # Transform on points
        if not self.on_manifold and isinstance(y, torch.Tensor):
            return self.Act(x, y)
        # scalar * manifold
        if self.on_manifold and not isinstance(y, LieGroup):
            if isinstance(y, torch.Tensor):
                assert y.dim()==0 or y.shape[-1]==1, "Tensor Dimension Invalid"
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
        return LieGroup(out, gtype=a.gtype)

    def AdjT(self, X, a):
        ''' Exp(a) * X = X * Exp(AdjT) '''
        if self.on_manifold:
            raise AttributeError("Gtype has no AdjT attribute")
        assert not X.gtype.on_manifold and a.gtype.on_manifold, "Gtype Invalid"
        assert X.gtype.group == a.gtype.group, "Gtype Invalid"
        out = self.__op__(self.group, adjT, X, a)
        return LieGroup(out, gtype=a.gtype)

    def Jinv(self, X, a):
        if self.on_manifold:
            raise AttributeError("Gtype has no Jinv attribute")
        return self.__op__(self.group, jinv, X, a)

    def matrix(self, gtensor):
        """ To 4x4 matrix """
        X = gtensor.Exp() if self.on_manifold else gtensor
        I = torch.eye(4, dtype=X.dtype, device=X.device)
        I = I.view([1] * (X.dim() - 1) + [4, 4])
        return X.unsqueeze(-2).Act(I).transpose(-1,-2)

    def translation(self, gtensor):
        """ To translation """
        X = gtensor.Exp() if self.on_manifold else gtensor
        p = torch.tensor([0., 0., 0.], dtype=X.dtype, device=X.device)
        return X.Act(p.view([1] * (X.dim() - 1) + [3,]))

    def quaternion(self, gtensor):
        raise NotImplementedError('quaternion not implemented yet')

    @classmethod
    def identity(cls, *args, **kwargs):
        raise NotImplementedError("Instance has no identity.")

    @classmethod
    def identity_like(cls, *args, **kwargs):
        return cls.identity(*args, **kwargs)

    def randn_like(self, *args, sigma=1, **kwargs):
        return self.randn(*args, sigma=1, **kwargs)

    def randn(self, *args, sigma=1., **kwargs):
        return sigma * torch.randn(*(tuple(args)+self.manifold), **kwargs)

    @classmethod
    def __op__(cls, group, op, x, y=None):
        inputs, out_shape = cls.__broadcast_inputs(x, y)
        out = op.apply(group, *inputs)
        dim = -1 if out.nelement() != 0 else x.shape[-1]
        return out.view(out_shape + (dim,))

    @classmethod
    def __broadcast_inputs(self, x, y):
        """ Automatic broadcasting of missing dimensions """
        if y is None:
            xs, xd = x.shape[:-1], x.shape[-1]
            return (x.view(-1, xd).contiguous(), ), x.shape[:-1]
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
    def cumprod(self, X, dim):
        return cumprod(X, dim)

    @classmethod
    def cumops_(self, X, dim, ops):
        return cumops_(X, dim, ops)

    @classmethod
    def cummul_(self, X, dim):
        return cummul_(X, dim)

    @classmethod
    def cumprod_(self, X, dim):
        return cumprod_(X, dim)


class SO3Type(GroupType):
    def __init__(self):
        super().__init__(1, 4, 4, 3)

    def Log(self, X):
        x = self.__op__(self.group, log, X)
        return LieGroup(x, gtype=so3_type)

    @classmethod
    def identity(cls, *args, **kwargs):
        data = torch.tensor([0., 0., 0., 1.], **kwargs)
        return LieGroup(data.expand(args+(-1,)), gtype=SO3_type)

    def randn(self, *args, sigma=1, requires_grad=False, **kwargs):
        data = so3_type.Exp(so3_type.randn(*args, sigma=sigma, **kwargs)).detach()
        return LieGroup(data, gtype=SO3_type).requires_grad_(requires_grad)

    def matrix(self, X):
        """ To 3x3 matrix """
        I = torch.eye(3, dtype=X.dtype, device=X.device)
        I = I.view([1] * (X.dim() - 1) + [3, 3])
        return X.unsqueeze(-2).Act(I).transpose(-1,-2)

    def identity_(self, X):
        X.fill_(0)
        X.index_fill_(dim=-1, index=torch.tensor([-1], device=X.device), value=1)
        return X


class so3Type(GroupType):
    def __init__(self):
        super().__init__(1, 3, 4, 3)

    def Exp(self, x):
        X = self.__op__(self.group, exp, x)
        return LieGroup(X, gtype=SO3_type)

    @classmethod
    def identity(cls, *args, **kwargs):
        return SO3_type.Log(SO3_type.identity(*args, **kwargs))

    def randn(self, *args, sigma=1, requires_grad=False, **kwargs):
        data = super().randn(*args, sigma=sigma, **kwargs).detach()
        return LieGroup(data, gtype=so3_type).requires_grad_(requires_grad)

    def matrix(self, gtensor):
        """ To 3x3 matrix """
        X = gtensor.Exp()
        I = torch.eye(3, dtype=X.dtype, device=X.device)
        I = I.view([1] * (X.dim() - 1) + [3, 3])
        return X.unsqueeze(-2).Act(I).transpose(-1,-2)

    def Jr(self, x):
        """
        Right jacobian of SO(3)
        The code is taken from the Sophus codebase :
        https://github.com/XueLianjie/BA_schur/blob/3af9a94248d4a272c53cfc7acccea4d0208b77f7/thirdparty/Sophus/sophus/so3.hpp#L113
        """
        I = torch.eye(3, device=x.device, dtype=x.dtype).expand(x.shape[:-1]+(3,3))
        theta = torch.linalg.norm(x)
        if theta < 1e-5:
            return I
        K = vec2skew(torch.nn.functional.normalize(x, dim=-1))
        return I - (1-theta.cos())/theta**2 * K + (theta - theta.sin())/torch.linalg.norm(x**3) * K@K


class SE3Type(GroupType):
    def __init__(self):
        super().__init__(3, 7, 7, 6)

    def Log(self, X):
        x = self.__op__(self.group, log, X)
        return LieGroup(x, gtype=se3_type)

    @classmethod
    def identity(cls, *args, **kwargs):
        data = torch.tensor([0., 0., 0., 0., 0., 0., 1.], **kwargs)
        return LieGroup(data.expand(args+(-1,)), gtype=SE3_type)

    def randn(self, *args, sigma=1, requires_grad=False, **kwargs):
        data = se3_type.Exp(se3_type.randn(*args, sigma=sigma, **kwargs)).detach()
        return LieGroup(data, gtype=SE3_type).requires_grad_(requires_grad)


class se3Type(GroupType):
    def __init__(self):
        super().__init__(3, 6, 7, 6)

    def Exp(self, x):
        X = self.__op__(self.group, exp, x)
        return LieGroup(X, gtype=SE3_type)

    @classmethod
    def identity(cls, *args, **kwargs):
        return SE3_type.Log(SE3_type.identity(*args, **kwargs))

    def randn(self, *args, sigma=1, requires_grad=False, **kwargs):
        data = super().randn(*args, sigma=sigma, **kwargs).detach()
        return LieGroup(data, gtype=se3_type).requires_grad_(requires_grad)


class Sim3Type(GroupType):
    def __init__(self):
        super().__init__(4, 8, 8, 7)

    def Log(self, X):
        x = self.__op__(self.group, log, X)
        return LieGroup(x, gtype=sim3_type)

    @classmethod
    def identity(cls, *args, **kwargs):
        data = torch.tensor([0., 0., 0., 0., 0., 0., 1., 1.], **kwargs)
        return LieGroup(data.expand(args+(-1,)), gtype=Sim3_type)

    def randn(self, *args, sigma=1, requires_grad=False, **kwargs):
        data = sim3_type.Exp(sim3_type.randn(*args, sigma=sigma, **kwargs)).detach()
        return LieGroup(data, gtype=Sim3_type).requires_grad_(requires_grad)


class sim3Type(GroupType):
    def __init__(self):
        super().__init__(4, 7, 8, 7)

    def Exp(self, x):
        X = self.__op__(self.group, exp, x)
        return LieGroup(X, gtype=Sim3_type)

    @classmethod
    def identity(cls, *args, **kwargs):
        return Sim3_type.Log(Sim3_type.identity(*args, **kwargs))

    def randn(self, *args, sigma=1, requires_grad=False, **kwargs):
        data = super().randn(*args, sigma=sigma, **kwargs).detach()
        return LieGroup(data, gtype=sim3_type).requires_grad_(requires_grad)


class RxSO3Type(GroupType):
    def __init__(self):
        super().__init__(2, 5, 5, 4)

    def Log(self, X):
        x = self.__op__(self.group, log, X)
        return LieGroup(x, gtype=rxso3_type)

    @classmethod
    def identity(cls, *args, **kwargs):
        data = torch.tensor([0., 0., 0., 1., 1.], **kwargs)
        return LieGroup(data.expand(args+(-1,)), gtype=rxso3_type)

    def randn(self, *args, sigma=1, requires_grad=False, **kwargs):
        data = rxso3_type.Exp(rxso3_type.randn(*args, sigma=sigma, **kwargs)).detach()
        return LieGroup(data, gtype=RxSO3_type).requires_grad_(requires_grad)


class rxso3Type(GroupType):
    def __init__(self):
        super().__init__(2, 4, 5, 4)

    def Exp(self, x):
        X = self.__op__(self.group, exp, x)
        return LieGroup(X, gtype=RxSO3_type)

    @classmethod
    def identity(cls, *args, **kwargs):
        return RxSO3_type.Log(RxSO3_type.identity(*args, **kwargs))

    def randn(self, *args, sigma=1, requires_grad=False, **kwargs):
        data = super().randn(*args, sigma=sigma, **kwargs).detach()
        return LieGroup(data, gtype=rxso3_type).requires_grad_(requires_grad)


SO3_type, so3_type = SO3Type(), so3Type()
SE3_type, se3_type = SE3Type(), se3Type()
Sim3_type, sim3_type = Sim3Type(), sim3Type()
RxSO3_type, rxso3_type = RxSO3Type(), rxso3Type()


class LieGroup(torch.Tensor):
    r""" The Base Class for LieGroup Tensor

    Returns:
        LieGroup Tensor (Inherited from torch.Tensor)

    Args:
        data (tensor): Data of Liegroup Tensor
        gtype (gtype): Group Type, which can be **Selected Below**:

    .. list-table:: List of **gtype**
        :widths: 25 25 30 30 30
        :header-rows: 1

        * - Name
          - Embedding
          - Alias Class
          - Manifold
          - Alias Class
        * - Orthogonal Group
          - :code:`SO3_type`
          - :meth:`SO3`
          - :code:`so3_type`
          - :meth:`so3`
        * - Euclidean Group
          - :code:`SE3_type`
          - :meth:`SE3`
          - :code:`se3_type`
          - :meth:`se3`
        * - Similarity Group
          - :code:`Sim3_type`
          - :meth:`Sim3`
          - :code:`sim3_type`
          - :meth:`sim3`
        * - Scaling Orthogonal
          - :code:`RxSO3_type`
          - :meth:`RxSO3`
          - :code:`rxso3_type`
          - :meth:`rxso3`

    Examples:
        >>> import torch
        >>> import pypose as pp
        >>> data = torch.randn(3, 3, requires_grad=True, device='cuda:0')
        >>> pp.LieGroup(data, gtype=pp.so3_type)
        so3Type Group:
        tensor([[ 0.9520,  0.4517,  0.5834],
                [-0.8106,  0.8197,  0.7077],
                [-0.5743,  0.8182, -1.2104]], device='cuda:0', grad_fn=<AliasBackward0>)

        >>> pp.so3(data)  ## Alias is Recommended
        so3Type Group:
        tensor([[ 0.9520,  0.4517,  0.5834],
                [-0.8106,  0.8197,  0.7077],
                [-0.5743,  0.8182, -1.2104]], device='cuda:0', grad_fn=<AliasBackward0>)

    Note:
        Alias of LieGroup for specific gtype is recommended.
        For example, the following usage is equivalent:

        - :code:`pp.LieGroup(data, gtype=pp.so3_type)`

        - :code:`pp.so3(data)`

    Note:
        All properties of :code:`torch.tensor` is available.
        See more details at https://pytorch.org/docs/stable/tensor_attributes.html

    """
    def __init__(self, data:torch.Tensor, gtype:GroupType):
        assert data.shape[-1:] == gtype.dimension, 'Dimension Invalid.'
        self.gtype = gtype

    def __new__(cls, data, gtype):
        return torch.Tensor.as_subclass(data, LieGroup)

    def __repr__(self):
        if hasattr(self, 'gtype'):
            return self.gtype.__class__.__name__ + " Group:\n" + super().__repr__()
        else:
            return super().__repr__()

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs={}):
        gtypes = (torch.Tensor if t is LieGroup or Parameter else t for t in types)
        data = torch.Tensor.__torch_function__(func, gtypes, args, kwargs)
        if data is not None and func.__name__ in HANDLED_FUNCTIONS:
            liegroup = args
            while not isinstance(liegroup, LieGroup):
                liegroup = liegroup[0]
            if isinstance(data, (tuple, list)):
                return (LieGroup(item, gtype=liegroup.gtype)
                        if isinstance(item, torch.Tensor) else item for item in data)
            return LieGroup(data, gtype=liegroup.gtype)
        return data

    @property
    def gshape(self) -> torch.Size:
        r'''
        LieGroup Tensor Shape (shape of torch.Tensor by removing the last dimension)

        Returns:
            torch.Size

        Note:
            - The only difference from :meth:`tensor.shape` is the last dimension is hidden.

            - The last dimension can also be accessed via :code:`LieGroup.gtype.dimension`.

        Examples:
            >>> x = pp.randn_SE3(2)
            >>> x.gshape
            torch.Size([2])
            >>> x.shape
            torch.Size([2, 7])
            >>> x.gtype.dimension
            torch.Size([7])
        '''
        return self.shape[:-1]

    def gview(self, *shape):
        r'''
        Args:
            shape (torch.Size or int...): the desired size

        Returns:
            A new lieGroup tensor sharing with the same data as the self tensor but of a different shape.

        Note:
            The only difference from :meth:`tensor.view` is the last dimension is hidden.

        Examples:
            >>> x = pp.randn_so3(2,2)
            >>> x.shape
            torch.Size([2, 2, 3])
            >>> x.gview(-1).gshape
            torch.Size([4])
        '''
        return self.view(*shape+self.gtype.dimension)

    def Exp(self):
        r'''
        See :meth:`pypose.Exp`
        '''
        return self.gtype.Exp(self)

    def Log(self):
        r'''
        See :meth:`pypose.Log`
        '''
        return self.gtype.Log(self)

    def Inv(self):
        return self.gtype.Inv(self)

    def Act(self, p):
        return self.gtype.Act(self, p)

    def __mul__(self, other):
        return self.gtype.Mul(self, other)

    def __matmul__(self, other):
        if isinstance(other, LieGroup):
            return self.gtype.Mul(self, other)
        else: # Same with: self.gtype.matrix(self) @ other
            return self.Act(other)

    def Retr(self, a):
        return self.gtype.Retr(self, a)

    def Adj(self, a):
        return self.gtype.Adj(self, a)

    def AdjT(self, a):
        return self.gtype.AdjT(self, a)

    def Jinv(self, a):
        return self.gtype.Jinv(self, a)

    def Jr(self):
        return self.gtype.Jr(self)

    def tensor(self) -> torch.Tensor:
        r'''
        Return the torch.Tensor without changing data.

        Return:
            Tensor: the torch.Tensor form of LieGroup Tensor.

        Example:
            >>> x = pp.randn_SO3(2)
            >>> x.tensor()
            tensor([[ 0.1196,  0.2339, -0.6824,  0.6822],
                    [ 0.9198, -0.2704, -0.2395,  0.1532]])
        '''
        return self.data

    def matrix(self) -> torch.Tensor:
        r'''
        Return LieGroup Tensor into matrix form.

        Return:
            Tensor: the batched matrix form (torch.Tensor) of LieGroup Tensor.

        Example:
            >>> x = pp.randn_SO3(2)
            >>> x.matrix()
            tensor([[[ 0.9285, -0.0040, -0.3713],
                    [ 0.2503,  0.7454,  0.6178],
                    [ 0.2743, -0.6666,  0.6931]],
                    [[ 0.4805,  0.8602, -0.1706],
                    [-0.7465,  0.2991, -0.5944],
                    [-0.4603,  0.4130,  0.7858]]])
        '''
        return self.gtype.matrix(self)

    def translation(self) -> torch.Tensor:
        r'''
        Extract the translation vector from a LieGroup Tensor.

        Return:
            Tensor: the batched translation.

        Example:
            >>> x = pp.randn_SE3(2)
            >>> x.translation()
            tensor([[-0.5358, -1.5421, -0.7224],
                    [ 0.8331, -1.4412,  0.0863]])
        '''
        return self.gtype.translation(self)

    def quaternion(self) -> torch.Tensor:
        return self.gtype.quaternion(self)

    def identity_(self):
        r'''
        Inplace set the LieGroup Tensor to identity.

        Return:
            LieGroup: the :code:`self` LieGroup Tensor

        Note:
            The translation part, if there is, is set to zeros, while the
            rotation part is set to identity quaternion.

        Example:
            >>> x = pp.randn_SO3(2)
            >>> x
            SO3Type Group:
            tensor([[-0.0724,  0.1970,  0.0022,  0.9777],
                    [ 0.3492,  0.4998, -0.5310,  0.5885]])
            >>> x.identity_()
            SO3Type Group:
            tensor([[0., 0., 0., 1.],
                    [0., 0., 0., 1.]])
        '''
        return self.gtype.identity_(self)

    def cumops(self, dim, ops):
        r"""
        See :func:`pypose.cumops`
        """
        return self.gtype.cumops(self, other, dim, ops)

    def cummul(self, dim):
        r"""
        See :func:`pypose.cummul`
        """
        return self.gtype.cummul(self, dim)

    def cumprod(self, dim):
        r"""
        See :func:`pypose.cumprod`
        """
        return self.gtype.cumprod(self, dim)

    def cumops_(self, dim, ops):
        r"""
        Inplace version of :func:`pypose.cumops`
        """
        return self.gtype.cumops_(self, other, dim, ops)

    def cummul_(self, dim):
        r"""
        Inplace version of :func:`pypose.cummul`
        """
        return self.gtype.cummul_(self, dim)

    def cumprod_(self, dim):
        r"""
        Inplace version of :func:`pypose.cumprod`
        """
        return self.gtype.cumprod_(self, dim)


class Parameter(LieGroup, nn.Parameter):
    r'''
    A kind of LieGroup Tensor that is to be considered a module parameter.

    Parameters are of :meth:`LieGroup` and :meth:`torch.nn.Parameter`,
    that have a very special property when used with Modules: when
    they are assigned as Module attributes they are automatically
    added to the list of its parameters, and will appear e.g., in
    :meth:`parameters()` iterator.

    Args:
        data (LieGroup): parameter LieGroup tensor.
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
        self.gtype = data.gtype

    def __new__(cls, data=None, requires_grad=True):
        if data is None:
            data = torch.tensor([])
        return LieGroup._make_subclass(cls, data, requires_grad)
