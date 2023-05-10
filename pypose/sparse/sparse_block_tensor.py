from typing import List, Optional
import torch
from torch import jit

'''Sparse Block Tensor for PyPose.

This module implements the sparse block tensor (referred to as SBT) for PyPose. SBT is designed
to be primarily used to represent sparse block matrices. We make extensive use of PyTorch's
sparse tensor for the current implementation, especially the COO tensor and the hybrid COO
tensor (referred to as Hybrid tensor). Some highlights:

* A block is a 2-dimensional dense tensor. So the dimension of an SBT is [ sH, sW, bH, bW ]
  where sH and sW are the sparse shape and bH and bW are block shape. See the following for the
  terminology.
* An SBT consists of a Hybrid tensor to represent the data storage and a plain COO tensor to
  preserve the block structure. We term them as the Storage and Proxy, respectively.
* All operations are implemented by using the PyTorch sparse tensor API. On high level, we
  convert a Hybrid tensor to the COO format, perform the operation by PyTorch, and then convert
  the result back to the Hybrid format.
* For all supported operations, we try to handle the Proxy correctly preserving the block
  structure where applicable.
* Gradient is supported where underlying PyTorch's operation supports it.

The following terminology is specific to our SBT implementation:

* block: A 2-dimensional dense tensor.
* block shape, dense shape, dense dimension: The shape of a single block.
* sparse shape, sparse dimension: The shape of the sparse tensor without irrespect to the block
  shape. Thus for a 3x3 SBT with block shape 2x2, the sparse shape is 3x3. The sparse shape of
  this SBT's COO equivalent is 6x6.
* non-zero structure: The block pattern of a Hybrid tensor or a non-zero pattern of a COO tensor.

For PyTorhc's sparse tensor, please refer to `Pytorch Sparse tensor`_.

Example:
    Examples to be added

        $ Awesome_exmaples.py

Attributes:
    module_level_variable1 (int): Module level variables.

.. _PyTorch Sparse tensor:
   https://pytorch.org/docs/stable/sparse.html#torch-sparse

'''

DENSE_SAFE_PROXY_THRES = 0


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
    dims = len(shape)
    out = torch.empty((dims, index.shape[0]), dtype=index.dtype, device=index.device)
    for dim_idx in range(dims-1, 0-1, -1):
        dim = shape[dim_idx]
        out[dim_idx] = (index % dim)
        index = index // dim
    return out


def prod(xs: List[int]) -> int:
    '''Compute the product of a list of integers.

    Args:
        l (list of int): The list of integers.

    Returns:
        int: The product of the list.
    '''
    p = 1
    for x in xs:
        p *= x
    return p


@jit.script
def make_coo_indices_and_dims_from_hybrid(hybrid):
    '''Create index and dimension info for converting a Hybrid tensor to a COO tensor.

    Create the index and dimension information for converting a Hybrid tensor to a COO tensor.

    Args:
        hybrid (torch.Tensor): The Hybrid tensor.

    Returns:
        torch.Tensor: The indices for creating a COO tensor.
        list of int: The dimension of the COO tensor.
    '''
    # Get the coalesced version such that we can operate on the indices.
    hybrid = hybrid.coalesce()

    # The block dimension.
    assert len(hybrid.shape) % 2 == 0, 'The hybrid tensor must have even number of dims, ' \
                                        'but got {}'.format(len(hybrid.shape))
    assert hybrid.dim() >= 4, f'hybrid should have dims >= 4, but got {hybrid.dim()}'
    num_dim = len(hybrid.shape) // 2  # Total number of the sparse dimensions.
    shape_b = hybrid.shape[num_dim:]  # Desne/block shape.
    shape_p = hybrid.shape[:num_dim]  # Sparse/proxy shape.
    numel_b = prod(shape_b)  # Number of elements per block.

    # === Compose target coo indices. ===

    # Index shift in block
    # the same as torch.cartesian_prod(*[torch.arange(i) for i in shape_b]).T
    index_shift = unravel_index(torch.arange(numel_b), shape_b)

    # Shift the original indices.
    indices_ori = hybrid.indices().unsqueeze(-1)  # (num_dim, num_b, 1)
    indices_rp = indices_ori.expand(-1, -1, numel_b)  # (num_dim, num_b, numel_b)
    # change to expand saves 0.3s for 10000 runs.

    # Compute the new indices by multiplying the block shape and adding the index shift.
    index_scale = torch.tensor(shape_b, dtype=torch.int64, device=hybrid.device)
    # index_scale, index_shift, indices_rp are all used in the form:
    # (num_dim, num_b, numel_b)
    indices_new = indices_rp * index_scale[:, None, None] + index_shift[:, None, :]
    indices_new = indices_new.flatten(1, 2)

    # === The sparse dimension of the target coo matrix. ===
    shape_coo: List[int] = []
    for i in range(num_dim):
        shape_coo.append(shape_p[i] * shape_b[i])
    return indices_new, shape_coo


@jit.script
def repeated_value_as_hybrid_value(coo: torch.Tensor,
                                   shape_b: List[int],
                                   mode: str = 'zeros',
                                   dtype: Optional[torch.dtype] = None,
                                   ):
    '''Create a Hybrid tensor based on a COO tensor and the mode value argument.
    The resulting hybrid tensor will have the same spatial structure as coo:
    any non-zero element of coo indicates existence of a block of size shape_b.
    Each block is filled by repeating an element value generated by the mode argument.

    For example, when setting `mode` to `"sequence"`, it create a Hybrid tensor based
    spatial structure specified by COO tensor with element value being block sequence
    number. When setting `mode` to `"zeros"`, it create a Hybrid tensor with all zero
    elements.

    Args:
        coo (torch.Tensor): The COO tensor.
        shape_b (List[int]): The block shape of the output Hybrid tensor.
        mode (str): Value to be repeated.
        dtype (torch.dtype): dtype of the created tensor.

    Returns:
        Hybrid tensor (torch.Tensor): The created Hybrid tensor.

    Example:
        # TODO
    '''
    coo = coo.coalesce()
    num_b = coo.values().shape[0]

    if mode == 'zeros':
        val = torch.zeros(num_b, device=coo.device, dtype=dtype)
    elif mode == 'ones':
        val = torch.ones(num_b, device=coo.device, dtype=dtype)
    elif mode == 'sequence':
        val = torch.arange(num_b, device=coo.device, dtype=dtype)
    else:
        raise ValueError(f'Unknown mode: {mode}')
    val = val[..., None, None].tile(shape_b)

    shape_coo: List[int] = list(coo.shape)

    return torch.sparse_coo_tensor(
            coo.indices(), val, size=shape_coo + shape_b
        ).coalesce()


@jit.script
def hybrid_2_coo(hybrid):
    '''Covnert a Hybrid tensor to a COO tensor.

    This function converts a Hybrid tensor to a COO tensor.

    Args:
        hybrid (torch.Tensor): The Hybrid tensor.

    Returns:
        COO tensor (torch.Tensor): The created COO tensor.

    '''
    hybrid = hybrid.coalesce()
    indices, shape_coo = make_coo_indices_and_dims_from_hybrid(hybrid)
    return torch.sparse_coo_tensor(indices, hybrid.values().flatten(), size=shape_coo)

@jit.script
def coo_2_hybrid(coo, proxy, dense_proxy_thres: int=DENSE_SAFE_PROXY_THRES):
    '''Convert a COO tensor to a Hybrid tensor by referring to the proxy.

    A proxy is a COO tensor. Any non-zero element in proxy indicates a block of the
    Hybrid tensor.

    Args:
        coo (torch.Tensor): The COO tensor.
        proxy (torch.Tensor): The proxy tensor as COO format.

    Returns:
        Hybrid tensor (torch.Tensor): The created Hybrid tensor.
    '''
    assert coo.device == proxy.device, \
        f'coo and proxy must be on the same device. '\
        f'coo.device = {coo.device}, proxy.device = {proxy.device}. '

    proxy = proxy.coalesce()

    # Figure out the shape of the target Hybrid tensor.
    shape_p = list(proxy.shape)
    assert all(coo.shape[i] % shape_p[i] == 0 for i in range(proxy.dim())), \
        f'coo and proxy shape are not compatible: coo.shape={coo.shape}, shape_p={shape_p}. '
    shape_b: List[int] = [coo.shape[i] // shape_p[i] for i in range(proxy.dim())]

    # Create a temporary Hybrid tensor to represent the block sequence.
    coo = coo.coalesce()
    # in-block indices.
    shape_b_t = torch.tensor(shape_b, dtype=torch.int64, device=coo.device).unsqueeze(-1)
    inblock_indices = coo.indices() % shape_b_t
    # block indices.
    indices_b = coo.indices() // shape_b_t
    numel_p = torch.numel(proxy)
    if numel_p < dense_proxy_thres:  # check the *dense* shape
        block_seq = torch.sparse_coo_tensor(
                proxy.indices(), torch.arange(proxy.values().shape[0]), size=proxy.shape
            )
        block_seq = block_seq.to_dense()  # dense is fast in indexing
        block_seq = block_seq[indices_b[0], indices_b[1]]
    else:
        # ravel multiple index into one.
        coeff = torch.tensor(shape_p+[1, ], dtype=shape_b_t.dtype, device=shape_b_t.device)
        coeff = coeff[1:].flipud().cumprod(dim=0).flipud()
        indices_p = (proxy.indices() * coeff.unsqueeze(-1)).sum(0, keepdim=True)
        block_seq = torch.sparse_coo_tensor(
                indices_p,
                torch.arange(proxy.values().shape[0]), size=(numel_p,)
            )
        block_seq = block_seq.index_select(
            0, (indices_b * coeff.unsqueeze(-1)).sum(0)).to_dense()

    # Index into a temporary tensor.
    blocks = torch.sparse_coo_tensor(
        indices=torch.cat([block_seq.unsqueeze(0), inblock_indices]),
        values=coo.values(),
        size=torch.Size([proxy.values().shape[0]] + shape_b)).to_dense()

    # Create the sparse hybrid COO tensor.
    return torch.sparse_coo_tensor(proxy.indices(), blocks, size=list(shape_p) + shape_b)


class SBTOperation(object):
    '''Sparse Block Tensor Operation base clase.

    An operation on an SBT will eventually be executed by the __torch_function__ method, where
    the type of SBT needs to be stripped and recovered in order to make PyTorch's dispatching
    mechanism work. Currently, an SBT consists of a Storage tensor and a Proxy tensor. All
    operations on an SBT must properly address the Proxy tensor. The type stripping-recovery
    cycle and the Proxy tensor handling are fundamentally associated with the operation itself.
    This SBTOperation class defines the interfaces for carrying out the above tasks.

    There are 4 methods defined in the SBTOperation class that the inheriting class can
    override. They are:

    * storage_pre: This method, most of the time, perform the type stripping operations. The
      Hybrid tensor to COO tensor conversion, if necessary, should also be implemented here.
    * storage_op: The actual operation performed on the Storage tensor.
    * proxy_op: The appropriate operation that needs to be performed on the Proxy tensor to
      preserve the block structure.
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
        # Forward all the arguments to PyTorch's __torch_function__ by default.
        return torch.Tensor.__torch_function__(func, stripped_types, s_args, kwargs)

    def proxy_op(self, func, stripped_types, p_args=(), kwargs={}):
        '''The operation on the Proxy tensor.

        Returns:
            tuple?

        '''
        # Do nothing about the Proxy tensor by default.
        return p_args

    def storage_post(self, func, types, s_outs=(), p_outs=(), kwargs={}):
        # Do nothing by default.
        return s_outs, p_outs

class SBTGetOp(SBTOperation):
    def __init__(self, func_name):
        super().__init__(func_name=func_name)

    def storage_pre(self, func, types, args=(), kwargs={}):
        '''Separate the Storange and Proxy tensor.

        Returns:
            s_array (list): A list of Storage tensors. Could be Hybrid tensors.
            p_array (list): A list of Proxy tensors.
        '''
        s_array = []
        p_array = []
        for arg in args:
            s_array.append( arg._s if isinstance(arg, SparseBlockTensor) else arg )
            p_array.append( arg._p if isinstance(arg, SparseBlockTensor) else arg )
        return s_array, p_array

class SBTProxyNoOp(SBTOperation):
    '''An SBT Operation that does not touch the Proxy tensor.

    This class implements the SBTOperation interfaces for the operations that do not touch the
    Proxy tensor of an SBT. Most of the time, such operations also work directly on the Storage
    tensor which is a Hybrid tensor. Therefore, the storage_pre and storage_post methods do not
    need to do the Hybrid-COO conversion.

    E.g., mostly the inplace operations such as torch.add_().

    Args:
        func_name (str): The name of the operation.

    '''

    def __init__(self, func_name):
        super().__init__(func_name=func_name)

    def storage_pre(self, func, types, args=(), kwargs={}):
        '''Separate the Storange and Proxy tensor.

        Returns:
            s_array (list): A list of Storage tensors. Could be Hybrid tensors.
            p_array (list): A list of Proxy tensors.
        '''
        s_array = []
        p_array = []
        for arg in args:
            s_array.append( arg._s if isinstance(arg, SparseBlockTensor) else arg )
            p_array.append( arg._p if isinstance(arg, SparseBlockTensor) else arg )
        return s_array, p_array

    # def storage_op(self, func, stripped_types, s_args=(), kwargs={}):
    #     # Forward all the arguments to parent's interface.
    #     return super().storage_op(func, stripped_types, s_args, kwargs)

    def proxy_op(self, func, stripped_types, p_args=(), kwargs={}):
        '''No-op on the Proxy tensor.

        Returns:
            p (torch.Tensor): The Proxy tensor.

        Note:
            Assume that the operation does not need to even touch the proxy tensor. For most of
            such operations, the proxy Tensor is the only sparse Tensor in the list of arguments.

        '''
        # Find the first sparse Tensor in operands.
        p = [ op for op in p_args
                if isinstance(op, torch.Tensor) and op.is_sparse == True
            ][0]
        return p

    def storage_post(self, func, types, s_outs=(), p_outs=(), kwargs={}):
        '''Pose processing on the Storage tensor.

        No need to do anything about the Storage and Proxy tensors.

        Returns:
            s_outs (tuple): A tuple of Storage tensors.
            p_outs (tuple): A tuple of Proxy tensors.

        Notes:
            s_outs (outs for Storage) and p_outs (outs for Proxy) are assumed to have the exact
            same order.

        '''
        return s_outs, p_outs

class SBTProxyCloneOp(SBTProxyNoOp):
    '''An SBT Operation that clones the Proxy tensor.

    This class implements the SBTOperation interfaces for the operations that clones the
    Proxy tensor of an SBT. Most of the time, such operations also work directly on the Storage
    tensor which is a Hybrid tensor. Therefore, the storage_pre and storage_post methods do not
    need to do the Hybrid-COO conversion.

    E.g., torch.abs().

    Args:
        func_name (str): The name of the operation.

    '''

    def __init__(self, func_name):
        super().__init__(func_name)

    def proxy_op(self, func, stripped_types, p_args=(), kwargs={}):
        '''Clone on the Proxy tensor.

        Returns:
            p (torch.Tensor): The Proxy tensor.

        Note:
            Assume that the operation does not need to even touch the proxy tensor. For most of
            such operations, the proxy Tensor is the only sparse Tensor in the list of
            arguments. The returned Proxy tensor is first detached from the input.

        '''
        # Find the first sparse Tensor in operands.
        p = [ op for op in p_args
                if isinstance(op, torch.Tensor) and op.is_sparse == True
            ][0]
        return p.detach().clone()

class SBTProxySameOpAsStorage(SBTOperation):
    '''SBT Operations that performs the same operation on the Proxy tensor.

    This class implements the SBTOperation that performs the same operation on the Proxy tensor.
    Most of such operations require to convert the Storage tensor from Hybrid to COO format.

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
        s_array = []
        p_array = []
        for arg in args:
            # Convert the sparse Hybrid tensor _s to COO tensor.
            s_array.append( hybrid_2_coo( arg._s )
                           if isinstance(arg, SparseBlockTensor)
                           else arg )

            # Do nothing about the Proxy tensor.
            p_array.append( arg._p if isinstance(arg, SparseBlockTensor) else arg )
        return s_array, p_array

    # def storage_op(self, func, stripped_types, s_args=(), kwargs={}):
    #     # Forward all the arguments to parent's interface.
    #     return super().storage_op(func, stripped_types, s_args, kwargs)

    def proxy_op(self, func, stripped_types, p_args=(), kwargs={}):
        '''Opertion on the Proxy tensor.

        Note:
            This method only gets called when the operation on the Storage returns sparse Tensor.
        '''
        return torch.Tensor.__torch_function__(func, stripped_types, p_args, kwargs)

    def storage_post(self, func, types, s_outs=(), p_outs=(), kwargs={}):
        '''Recover the block structure of s_outs.

        Returns:
            s_outs (list): A list of Storage tensors in Hybrid format.
            p_outs (tuple): A tuple of Proxy tensors.

        Note:
            s_outs (outs for Storage) and p_outs (outs for Proxy) are assumed to have the exact
            same order.
        '''
        s_outs = [ coo_2_hybrid(s, p)
                    if isinstance(s, torch.Tensor) and s.is_sparse == True
                    else s
                    for s, p in zip(s_outs, p_outs) ]

        return s_outs, p_outs

class OpType(object):
    '''A simple union of function name and operation type.

    Note:
        if func_name is None (or not provided), then the __name__ of a function is used.

    Args:
        op_type (SBTOperation): The type/class of the operation.
        func_name (str): The name of the operation used by __torch_function__. Optional.
    '''

    def __init__(self, op_type, func_name=None):
        super().__init__()
        self.op_type = op_type
        self.func_name = func_name

# HFS: Handled Functions by SBT.
class HFS:

    HANDLED_FUNCS = dict()
    '''dict: A dictionary of all supported operations on SparseBlockTensor.

    Key: The name of the operation.
    Value: An SBTOperation object.
    '''

    @classmethod
    def is_handled(cls, func_name):
        '''Check if a operation is supported by SBT.

        Returns:
            bool: True if the operation is supported by SBT. False otherwise.
        '''
        return func_name in cls.HANDLED_FUNCS

    @classmethod
    def add_op(cls, name, op_type):
        '''Add an operation to the supported operations on SBT.

        Args:
            name (str): The name of the operation.
            op_type (SBTOperation): The SBTOperation class.
        '''
        if name in cls.HANDLED_FUNCS:
            raise ValueError(f'The operation {name} is already registered. ')
        cls.HANDLED_FUNCS[name] = op_type(name)

    @classmethod
    def register(cls, sbt_op_type):
        '''Thie function is meant to be used as a decorator.

        Args:
            sbt_op_type (OpType): The OpType object describing the operation type and
                the function name.

        '''
        def decorator(func):
            if sbt_op_type.func_name is None:
                cls.add_op(func.__name__, sbt_op_type.op_type)
            else:
                cls.add_op(sbt_op_type.func_name, sbt_op_type.op_type)
            return func
        return decorator

# =========================================================================
# ========== Register supported operations through HFS directly. ==========
# =========================================================================

# Operations can be registered in two ways:
# 1. Register the operation directly through HFS.add_op().
# 2. Register the operation through the @HFS.register() decorator.
# The second way can be used if the operation is an override of a torch.Tensor method.

# ========== Special Python methods. ==========
# HFS.add_op( '__format__', SBTOperation )
HFS.add_op( '__get__', SBTGetOp )

# ========== Linear Algebra operations. ==========
# HFS.add_op( 'matmul', SBTProxySameOpAsStorage )

# ========== Tensor methods. ==========

# ========== Operations for COO tensors. ==========

# ========== Unary functions. ==========
HFS.add_op( 'abs', SBTProxyCloneOp )

# ==============================================================
# ========== End of supported operation registration. ==========
# ==============================================================

class SparseBlockTensor(torch.Tensor):

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs={}):
        '''The main entry point for all operations on SparseBlockTensor.'''

        if not HFS.is_handled(func.__name__):
            raise Exception(
                f'All operations on SparseBlockTensor must be handled. '
                f'\n{func.__name__} is not.')

        sbt_op = HFS.HANDLED_FUNCS[func.__name__]

        args_storage, args_proxy = sbt_op.storage_pre(func, types, args, kwargs)

        # Strip types
        stripped_types = (torch.Tensor if t is SparseBlockTensor else t for t in types)
        # if func.__name__ == 'matmul':
        #     import ipdb; ipdb.set_trace()
        #     pass

        # Let PyTorch do its dispatching.
        outputs_storage = sbt_op.storage_op(func, stripped_types, args_storage, kwargs)

        # Handle the proxy.
        outputs_proxy = sbt_op.proxy_op(func, stripped_types, args_proxy, kwargs)

        # Do post processing.
        if not isinstance(outputs_storage, (list, tuple)):
            flag_list = False
            outputs_storage = [outputs_storage]
        else:
            flag_list = True

        if not isinstance(outputs_proxy, (list, tuple)):
            outputs_proxy = [outputs_proxy]

        outputs_list_storage, outputs_list_proxy = \
            sbt_op.storage_post(func, stripped_types, outputs_storage, outputs_proxy, kwargs)

        if outputs_list_storage[0] is None:
            return None

        outputs_final = []
        for output_storage, output_proxy in zip( outputs_list_storage, outputs_list_proxy ):
            # Recover the types.
            if isinstance(output_storage, torch.Tensor) and not isinstance(output_storage, cls):
                if not output_storage.is_sparse:
                    outputs_final.append( output_storage )
                    continue

                # Wrap s and p into a SparseBlockTensor.
                sbt = cls()
                sbt._s = output_storage
                sbt._p = output_proxy
                outputs_final.append( sbt )
            # Not a tensor or not a SparseBlockTensor.
            outputs_final.append( outputs_storage )

        if flag_list:
            return outputs_final
        else:
            return outputs_final[0]

    def __repr__(self):
        r'''
        t = SparseBlockTensor()
        >>>t
        SparseBlockTensor()
        '''
        return str(self)

    def __str__(self):
        r'''
        t = SparseBlockTensor()
        print( t )
        ' SparseBlockTensor() '
        '''
        return f"SparseBlockTensor\nstorage:\n{self._s}\nproxy:\n{self._p}"

    @HFS.register(OpType(op_type=SBTOperation))
    def __format__(self, spec):
        return str(self)

    @HFS.register(OpType(op_type=SBTProxySameOpAsStorage, func_name='matmul'))
    def __matmul__(self, other):
        r'''
        return the corresponding sparse matrix and index matrix
        '''
        return torch.matmul(self, other)

    @HFS.register(OpType(op_type=SBTProxySameOpAsStorage, func_name='add'))
    def __add__(self, other):
        r'''
        return the corresponding sparse matrix and index matrix
        '''
        return torch.add(self, other)

    @HFS.register(OpType(op_type=SBTProxySameOpAsStorage, func_name='sub'))
    def __sub__(self, other):
        r'''
        return the corresponding sparse matrix and index matrix
        '''
        return torch.sub(self, other)
    # NOTE: for torch operations that need special treatment, place an override here. Then call
    # the corresponding function of PyTorch to begin the dispatching. E.g.:
    # def abs(self):
    #     print('This is SBT abs(). ')
    #     return torch.abs(self)


def sparse_block_tensor(
        indices, values, size=None, dtype=None, device=None, requires_grad=False):
    # Figure out the block shape.
    num_b, shape_b = values.shape[0], values.shape[1:]

    # Storage.
    storage = torch.sparse_coo_tensor(
        indices,
        values,
        size=(*size, *shape_b),
        dtype=dtype,
        device=device,
        requires_grad=requires_grad ).coalesce()

    proxy = torch.sparse_coo_tensor(
        indices,
        torch.ones(num_b, dtype=dtype, device=device),
        size=size,
        dtype=dtype,
        device=device,
        requires_grad=False ).coalesce()

    x = SparseBlockTensor()
    x._s = storage # s for storage.
    x._p = proxy
    return x

def test_sparse_coo_2_sparse_hybrid_coo():
    print()

    i = torch.tensor([
        [0, 0, 1, 1, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
        [0, 1, 0, 1, 4, 5, 4, 5, 2, 3, 2, 3, 4, 5, 4, 5] ], dtype=torch.int64)
    v = torch.arange(16).float()
    s = torch.sparse_coo_tensor(i, v, size=(6,6))

    i = torch.Tensor([
        [0, 0, 1, 2],
        [0, 2, 1, 2] ]).to(dtype=torch.int64)
    v = torch.ones(4, dtype=torch.int64)
    p = torch.sparse_coo_tensor(i, v, size=(3,3))

    print(f's = \n{s.to_dense()}')
    print(f'p = \n{p.to_dense()}')

    h = coo_2_hybrid(s, p)
    print(f'h = \n{h.to_dense()}')
