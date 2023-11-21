import torch
from torch import jit
from typing import List

'''Sparse Block Tensor (SbkTensor) for PyPose.
This module implements the sparse block tensor (referred to as SbkTensor) for PyPose.
'''


@jit.script
def ravel_multi_index(coords: torch.Tensor, shape: List[int]) -> torch.Tensor:
    """Converts a tensor of coordinate vectors into a tensor of flat indices.
    This function is not used in the current implementation, because instantiating
    a tensor is time-consuming (0.5s latency for 10000 runs).
    It is kept here for future reference.

    This is a `torch` implementation of `numpy.ravel_multi_index`.

    Args:
        coords: A tensor of coordinate vectors, (*, D).
        shape: The source shape.

    Returns:
        The raveled indices, (*,).
    """

    shape = torch.tensor(shape + [1,], dtype=coords.dtype, device=coords.device)
    coefs = shape[1:].flipud().cumprod(dim=0).flipud()

    return (coords * coefs.unsqueeze(-1)).sum(dim=0, keepdim=True)


def unravel_index(index, shape: List[int]):
    '''
    This is a `torch` implementation of `numpy.unravel_index`.
    Converts a flat index or array of flat indices into a tuple of coordinate arrays.
    Args:
        index (Tensor): An integer array whose elements are indices into the flattened
        version of an array of dimensions shape.
        shape (List[int]): The shape of the array to use for unraveling index.
    Returns:
        Tensor: Each row in the tensor has the same shape as the index tensor.
        Each column in the tensor corresponds to the dimension in shape.
        Shape: (dim, numel)
    '''
    index = index.view(-1, 1)  # Ensure index is a column vector
    strides = torch.tensor(shape[::-1], device=index.device).cumprod(0).flip(dims=(0,))
    shape = torch.tensor(shape, device=index.device)
    return ((index % strides) // (strides // shape)).t()


@jit.script
def hybrid2coo(hybrid):
    '''Covnert a Hybrid tensor to a COO tensor.

    This function converts a Hybrid tensor to a COO tensor.

    Args:
        hybrid (torch.Tensor): The Hybrid tensor.

    Returns:
        COO tensor (torch.Tensor): The created COO tensor.
    '''
    hybrid = hybrid.coalesce()
    assert hybrid.dim() % 2 == 0, 'The hybrid tensor must have even number of dims, ' \
                                        'but got {}'.format(hybrid.dim())
    assert hybrid.dim() >= 4, f'hybrid should have dims >= 4, but got {hybrid.dim()}'
    dim = hybrid.dim() // 2         # sparse dimensions.
    blkshape = hybrid.shape[dim:]   # block shape.
    proxyshape = hybrid.shape[:dim] # proxy shape.
    blknum = hybrid.indices().shape[-1]
    blknumel = 0 if blknum == 0 else hybrid.values().numel() // blknum
    offset = unravel_index(torch.arange(blknumel), blkshape) # in-block offset
    # indices shape (dim, blknum, blknumel)
    indices = hybrid.indices().unsqueeze(-1).expand(-1, -1, blknumel)
    # scale the block indices by a factor of block shape
    scale = torch.tensor(blkshape, dtype=torch.int64, device=hybrid.device)
    # scale, offset, indices are all in the form: (dim, blknum, blknumel)
    indices = (indices * scale[:, None, None] + offset[:, None, :]).flatten(1, 2)
    shape = [proxyshape[i] * blkshape[i] for i in range(dim)]
    return torch.sparse_coo_tensor(indices, hybrid.values().flatten(), size=shape)


@jit.script
def coo2hybrid(coo, proxy, dense_proxy_limit: int=30000):
    '''Convert a COO tensor to a Hybrid tensor by referring to the proxy.

    A proxy is a COO tensor. Any non-zero element in proxy indicates a block of the
    Hybrid tensor.

    Args:
        coo (torch.Tensor): The COO tensor.
        proxy (torch.Tensor): The proxy tensor as COO format.
        dense_proxy_limit (int): a threshold to use dense blocks for faster indexing.
            Default: 30000.

    Returns:
        Hybrid tensor (torch.Tensor): The created Hybrid tensor.
    '''
    assert coo.device == proxy.device, \
        f'coo and proxy must be on the same device. '\
        f'coo.device = {coo.device}, proxy.device = {proxy.device}. '

    coo = coo.coalesce()
    proxy = proxy.coalesce()

    # Figure out the shape of the target Hybrid tensor.
    shape_p = list(proxy.shape)
    assert all(coo.shape[i] % shape_p[i] == 0 for i in range(proxy.dim())), \
        f'coo and proxy shape are incompatible: coo.shape={coo.shape}, shape_p={shape_p}.'
    shape_b: List[int] = [coo.shape[i] // shape_p[i] for i in range(proxy.dim())]

    shape_b_t = torch.tensor(shape_b, dtype=torch.int64, device=coo.device).unsqueeze(-1)
    offsets = coo.indices() % shape_b_t
    indices_b = coo.indices() // shape_b_t # block indices
    numel_p = torch.numel(proxy)  # dense number of elements
    values = torch.arange(proxy.values().shape[0])
    if numel_p < dense_proxy_limit:  # check the *dense* shape
        block_seq = torch.sparse_coo_tensor(proxy.indices(), values, size=proxy.shape)
        block_seq = block_seq.to_dense()  # dense is fast in indexing
        block_seq = block_seq[indices_b[0], indices_b[1]]
    else:
        # ravel multiple index into one.
        coeff = torch.tensor(shape_p+[1, ], dtype=torch.int64, device=coo.device)
        coeff = coeff[1:].flipud().cumprod(dim=0).flipud()
        indices_p = (proxy.indices() * coeff.unsqueeze(-1)).sum(0, keepdim=True)
        block_seq = torch.sparse_coo_tensor(indices_p, values, size=(numel_p,))
        select_index = (indices_b * coeff.unsqueeze(-1)).sum(0)
        block_seq = block_seq.index_select(0, select_index).to_dense()

    indices = torch.cat([block_seq.unsqueeze(0), offsets])
    size = torch.Size(values.shape + shape_b)
    blocks = torch.sparse_coo_tensor(indices, coo.values(), size).to_dense()

    return torch.sparse_coo_tensor(proxy.indices(), blocks, size=list(shape_p) + shape_b)


class SbkOps(object):
    '''Operation base class for SbkTensor

    An operation on an SbkTensor will eventually be executed by the __torch_function__
    method. An SbkTensor consists of a Storage tensor and a Proxy tensor. All
    operations must properly address the Proxy tensor. This Operation class defines the
    interfaces for carrying out the above tasks.

    There are 4 methods defined in the Operation class that the inheriting class can
    override. They are:

    * storage_pre: This method, most of the time, perform the type stripping operations.
    The Hybrid tensor to COO tensor conversion, if necessary, should be implemented here.
    * storage_op: The actual operation performed on the Storage tensor.
    * proxy_op: The appropriate operation that needs to be performed on the Proxy tensor
    to preserve the block structure.
    * storage_post: This method, most of the time, perform the type recovery operations.

    Args:
        func_name (str): The name of the operation.

    Attributes:
        func_name (str): The name of the operation.

    '''

    def __init__(self, func_name):
        super().__init__()
        self.func_name = func_name

    def storage_pre(self, func, types, args=(), kwargs={}):
        return args, args

    def storage_op(self, func, stripped_types, s_args=(), kwargs={}):
        return torch.Tensor.__torch_function__(func, stripped_types, s_args, kwargs)

    def proxy_op(self, func, stripped_types, p_args=(), kwargs={}):
        '''The operation on the Proxy tensor.

        Returns:
            tuple?

        '''
        return p_args

    def storage_post(self, func, types, s_outs=(), p_outs=(), kwargs={}):
        return s_outs, p_outs


class SbkGetOp(SbkOps):
    def __init__(self, func_name):
        super().__init__(func_name=func_name)

    def storage_pre(self, func, types, args=(), kwargs={}):
        '''Separate the Storage and Proxy tensor.

        Returns:
            s_array (list): A list of Storage tensors. Could be Hybrid tensors.
            p_array (list): A list of Proxy tensors.
        '''
        s_array = [arg._s if isinstance(arg, SbkTensor) else arg for arg in args]
        p_array = [arg._p if isinstance(arg, SbkTensor) else arg for arg in args]
        return s_array, p_array


class HybridOps(SbkOps):
    '''An Operation that does not need to disassemble the Storage tensor.
    E.g., mostly the inplace operations such as torch.add_().

    Args:
        func_name (str): The name of the operation.
        proxy_reduction (str): The reduction operation on the Proxy tensor that produces
            the Proxy tensor for result. E.g., 'add' for torch.add(). 'mul' for
            torch.mul(). If None, then the Proxy tensor is the only sparse tensor in the
            list of arguments.
        clone (bool): If True, then clone the Proxy tensor before applying the
            reduction operation.
    '''

    def __init__(self, func_name, proxy_reduction=None, clone=False):
        super().__init__(func_name=func_name)
        self.proxy_reduction = proxy_reduction
        self.clone = clone

    def storage_pre(self, func, types, args=(), kwargs={}):
        '''Separate the Storange and Proxy tensor.

        Returns:
            s_array (list): A list of Storage tensors. Could be Hybrid tensors.
            p_array (list): A list of Proxy tensors.
        '''
        s_array = [arg._s if isinstance(arg, SbkTensor) else arg for arg in args]
        p_array = [arg._p if isinstance(arg, SbkTensor) else arg for arg in args]
        return s_array, p_array

    def proxy_op(self, func, stripped_types, p_args=(), kwargs={}):
        '''Apply operation on the Proxy tensor, according to the proxy_reduction.

        Returns:
            p (torch.Tensor): The Proxy tensor.

        Note:
            Assume that the operation does not need to even touch the proxy tensor.
            For most of such operations, the proxy Tensor is the only sparse Tensor in
            the list of arguments.
        '''
        # Find the first sparse Tensor in operands.
        if self.proxy_reduction is None:
            func = lambda op: isinstance(op, torch.Tensor) and op.is_sparse
            p = next(filter(func, p_args), None)
        elif self.proxy_reduction == 'add':
            p = torch.add(*p_args)
        elif self.proxy_reduction == 'mul':
            p = torch.mul(*p_args)
        else:
            raise ValueError('Unknown proxy reduction: {}'.format(self.proxy_reduction))
        if self.clone:
            return p.detach().clone()
        return p


class CooOps(SbkOps):
    '''SbkOps that performs the same operation on the Proxy tensor.

    This class implements the SbkOps that performs the same operation on the Proxy tensor.
    Most of such operations require converting the Storage tensor from Hybrid to COO format.

    E.g., torch.add().

    Args:
        func_name (str): The name of the operation.

    '''

    def __init__(self, func_name):
        super().__init__(func_name)

    def storage_pre(self, func, types, args=(), kwargs={}):
        '''Separate the Storage and Proxy tensors. Convert the Storage tensor to COO format.

        Returns:
            s_array (list): A list of Storage tensors that are converted to COO format.
            p_array (list): A list of Proxy tensors.
        '''
        s_array = [hybrid2coo(arg._s) if isinstance(arg, SbkTensor) else arg for arg in args]
        p_array = [arg._p if isinstance(arg, SbkTensor) else arg for arg in args]
        return s_array, p_array

    def proxy_op(self, func, stripped_types, p_args=(), kwargs={}):
        '''Opertion on the Proxy tensor.

        Note:
            This method only gets called when the operation on the Storage returns sparse
            Tensor.
        '''
        return torch.Tensor.__torch_function__(func, stripped_types, p_args, kwargs)

    def storage_post(self, func, types, s_outs=(), p_outs=(), kwargs={}):
        '''Recover the block structure of s_outs.

        Returns:
            s_outs (list): A list of Storage tensors in Hybrid format.
            p_outs (tuple): A tuple of Proxy tensors.

        Note:
            s_outs (outs for Storage) and p_outs (outs for Proxy) are assumed to have the
            exact same order.
        '''
        s_outs = [ coo2hybrid(s, p)
                    if isinstance(s, torch.Tensor) and s.is_sparse == True
                    else s
                    for s, p in zip(s_outs, p_outs) ]

        return s_outs, p_outs

class OpType(object):
    '''A simple union of function name and operation type.

    Note:
        if func_name is None (or not provided), then the __name__ of a function is used.

    Args:
        op_type (SbkOps): The type/class of the operation.
        func_name (str, optional): The name of the operation used by __torch_function__.
    '''

    def __init__(self, op_type, func_name=None):
        super().__init__()
        self.op_type = op_type
        self.func_name = func_name


class registry:
    HANDLED_FUNCS = dict()
    # Key: The name of the operation. Value: An SbkOps object.

    @classmethod
    def is_handled(cls, func_name):
        '''Check if a operation is supported by SbkTensor.

        Returns:
            bool: True if the operation is supported by SbkTensor. False otherwise.
        '''
        return func_name in cls.HANDLED_FUNCS

    @classmethod
    def add_op(cls, name, op_type, *args, **kwargs):
        '''Add an operation to the supported operations on SbkTensor.

        Args:
            name (str): The name of the operation.
            op_type (SbkOps): The SbkOps class.
        '''
        if name in cls.HANDLED_FUNCS:
            raise ValueError(f'The operation {name} is already registered. ')
        cls.HANDLED_FUNCS[name] = op_type(name, *args, **kwargs)

    @classmethod
    def register(cls, op_type, *args, **kwargs):
        '''Thie function is meant to be used as a decorator.

        Args:
            op_type (OpType): The OpType object describing the operation type and
                the function name.

        '''
        def decorator(func):
            func_name = func.__name__ if op_type.func_name is None else op_type.func_name
            cls.add_op(func_name, op_type.op_type, *args, **kwargs)
            return func
        return decorator

# Operations can be registered in two ways:
# 1. Register the operation directly through registry.add_op().
# 2. Register the operation through the @registry.register() decorator.

# ========== Special Python methods. ==========
registry.add_op( '__get__', SbkGetOp )

# ========== Linear Algebra operations. ==========
registry.add_op( 'matmul', CooOps )

# ========== elementwise functions ==========
registry.add_op('abs', HybridOps, proxy_reduction=None, clone=True)
registry.add_op('add', HybridOps, proxy_reduction='add')
registry.add_op('asin', HybridOps, proxy_reduction=None, clone=True)
registry.add_op('atan', HybridOps, proxy_reduction=None, clone=True)
registry.add_op('ceil', HybridOps, proxy_reduction=None, clone=True)
registry.add_op('floor', HybridOps, proxy_reduction=None, clone=True)
registry.add_op('round', HybridOps, proxy_reduction=None, clone=True)
registry.add_op('sin', HybridOps, proxy_reduction=None, clone=True)
registry.add_op('sinh', HybridOps, proxy_reduction=None, clone=True)
registry.add_op('sqrt', HybridOps, proxy_reduction=None, clone=True)
registry.add_op('square', HybridOps, proxy_reduction=None, clone=True)
registry.add_op('sub', HybridOps, proxy_reduction='add')
registry.add_op('tan', HybridOps, proxy_reduction=None, clone=True)
registry.add_op('tanh', HybridOps, proxy_reduction=None, clone=True)


class SbkTensor(torch.Tensor):

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs={}):
        '''The main entry point for all operations on SbkTensor.'''

        if not registry.is_handled(func.__name__):
            raise Exception(
                f'All operations on SbkTensor must be handled. '
                f'\n{func.__name__} is not.')

        op = registry.HANDLED_FUNCS[func.__name__]
        args_storage, args_proxy = op.storage_pre(func, types, args, kwargs)
        types = (torch.Tensor if t is SbkTensor else t for t in types)
        storages = op.storage_op(func, types, args_storage, kwargs)
        proxies = op.proxy_op(func, types, args_proxy, kwargs)
        if not isinstance(storages, (list, tuple)):
            flag_list = False
            storages = [storages]
        else:
            flag_list = True

        if not isinstance(proxies, (list, tuple)):
            proxies = [proxies]

        outs_storage, outs_proxy = op.storage_post(func, types, storages, proxies, kwargs)

        if outs_storage[0] is None:
            return None

        outputs_final = []
        for storage, proxy in zip(outs_storage, outs_proxy):
            # Recover the types
            if isinstance(storage, torch.Tensor) and not isinstance(storage, cls):
                if not storage.is_sparse:
                    outputs_final.append( storage )
                    continue
                sbk = cls()
                sbk._s = storage
                sbk._p = proxy
                outputs_final.append(sbk)
            else:
               # noop for the case of not a tensor nor a SbkTensor
                outputs_final.append(storage)

        if flag_list:
            return outputs_final
        else:
            return outputs_final[0]

    def __repr__(self):
        r'''
        t = SbkTensor()
        >>>t
        SbkTensor()
        '''
        return str(self)

    def __str__(self):
        r'''
        t = SbkTensor()
        print( t )
        ' SbkTensor() '
        '''
        return f"SbkTensor Containing:\nStorage:\n{self._s}\nProxy:\n{self._p}"

    @registry.register(OpType(op_type=SbkOps))
    def __format__(self, spec):
        return str(self)

    @registry.register(OpType(op_type=HybridOps, func_name='mul'))
    def __mul__(self, other):
        res = torch.mul(self, other)
        # dim auto correction
        s = res._s
        p = res._p
        if s.sparse_dim() != s.dense_dim() and s.indices().shape[1] == 0:
            num_dim = (s.sparse_dim() + s.dense_dim()) // 2
            shape_b = s.shape[num_dim:]
            assert p.dim() == num_dim
            assert p.shape == s.shape[:num_dim]
            indices = torch.tensor([]).reshape((num_dim, 0))
            values = torch.tensor([]).reshape(0, *shape_b)
            size = (*p.shape, *shape_b)
            res._s = torch.sparse_coo_tensor(indices, values, size).coalesce()

        return res

    def to_sparse_coo(self):
        '''Convert the SbkTensor to a COO tensor.

        Returns:
            torch.Tensor: The converted COO tensor.
        '''
        return hybrid2coo(self._s)

    def to_dense(self):
        return hybrid2coo(self._s).to_dense()

    def sparse_dim(self):
        '''Return the sparse dimension of the internal storage.'''
        return self._s.sparse_dim()

    def dense_dim(self):
        '''Return the dense dimension of the internal storage.'''
        return self._s.dense_dim()

    def indices(self):
        '''Return the indices of the internal storage.'''
        return self._s.indices()


def sbktensor(indices, values, size=None, dtype=None, device=None, requires_grad=False):
    # Figure out the block shape.
    num_b, shape_b = values.shape[0], values.shape[1:]
    assert len(indices) == len(shape_b), \
        f'sparse_dim = {len(indices)}, dense_dim = {len(shape_b)}. The two must be equal.'
    x = SbkTensor()

    # Storage.
    x._s = torch.sparse_coo_tensor(
        indices,
        values,
        size=(*size, *shape_b),
        dtype=dtype,
        device=device,
        requires_grad=requires_grad).coalesce()

    # Proxy.
    x._p = torch.sparse_coo_tensor(
        indices,
        torch.ones(num_b, dtype=dtype, device=device),
        size=size,
        dtype=dtype,
        device=device,
        requires_grad=False).coalesce()

    return x

def sparse_coo_diagonal(t: torch.Tensor):
    indices = t.indices()
    diag_indices = indices[0] == indices[1]
    return t.values()[diag_indices]

def sparse_coo_diagonal_clamp_(t: torch.Tensor, min_value, max_value):
    indices = t.indices()
    diag_indices = indices[0] == indices[1]
    t.values()[diag_indices] = t.values()[diag_indices].clamp_(min_value, max_value)


def sparse_coo_diagonal_add_(t: torch.Tensor, other: torch.Tensor):
    indices = t.indices()
    diag_indices = indices[0] == indices[1]
    t.values()[diag_indices] = t.values()[diag_indices].add_(other)
