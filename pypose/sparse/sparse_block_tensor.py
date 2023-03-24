
import torch
from torch.utils._pytree import tree_map, tree_flatten

def make_coo_indices_and_dims_from_hybrid(hybrid):
    # Get the coalesced version such that we can operate on the indices.
    hybrid = hybrid.coalesce()
    
    # The original tensor dimension and block dimension.
    t_dim, b_dim = hybrid.shape[:2], hybrid.shape[2:]
    assert len(b_dim) == 2, f'hybrid.shape = {hybrid.shape}. '
    n_block = hybrid.values().shape[0]
    n_block_elem = b_dim[0] * b_dim[1]
    
    # === Compose target coo indices. ===
    indices_ori = hybrid.indices()
    
    # Index shift for every element in a block.
    shift_row = torch.arange(b_dim[0], dtype=torch.int64, device=hybrid.device)
    shift_col = torch.arange(b_dim[1], dtype=torch.int64, device=hybrid.device)    
    index_shift_row, index_shift_col = torch.meshgrid( shift_row, shift_col, indexing='ij' )
    
    # Flatten the index shift.
    index_shift_row = index_shift_row.contiguous().view((-1,)) # contiguous() is necessary.
    index_shift_col = index_shift_col.contiguous().view((-1,))
    index_shift = torch.stack( (index_shift_row, index_shift_col), dim=0 ) # 2 * n_block_elem
    
    # Repeat and shift the original indices.
    indices_rp = indices_ori.repeat_interleave(n_block_elem, dim=1) # 2 * (n_block * n_block_elem)
    indices_rp = indices_rp.view((2, n_block, n_block_elem)) # 2 * n_block * n_block_elem
    indices_pm = indices_rp.permute(1, 0, 2) # n_block * 2 * n_block_elem
    
    index_scale = torch.Tensor([*b_dim]).to(dtype=torch.int64, device=hybrid.device)
    index_scale = index_scale.view((1, 2, 1))
    
    indices_new = indices_pm * index_scale + index_shift # n_block * 2 * n_block_elem
    indices_new = indices_new.permute((1, 0, 2)).view((2, -1)) # 2 * n_block * n_block_elem -> 2 * (n_block * n_block_elem)
    
    # === The dimension of the target coo matrix. ===
    coo_dim = [ t_dim[0]*b_dim[0], t_dim[1]*b_dim[1] ]
    
    return indices_new, coo_dim

def sparse_coo_2_hybrid_block_sequence(s, block_shape):
    '''
    s is a sparse COO tensor. Any non-zero element in s indicates a block of size block_shape.
    This function returns a new sparse hybrid COO tensor, which has the same block structure as s.
    Every block of the hybrid tensor has the block sequnce number as the value for all of its elements.
    '''
    
    # Make sure we have ordered values.
    s = s.coalesce()
    
    # Only use the sparse dimension.
    t_dim = s.shape[:2]
    
    # Number of block and number of elements per block.
    n_block = s.values().shape[0]
    n_block_elem = block_shape[0] * block_shape[1]
    
    # Prepare the sequence number.
    block_seq = torch.arange(n_block, dtype=torch.int64, device=s.device)
    block_seq = block_seq.repeat_interleave(n_block_elem).view((n_block, *block_shape))
    
    return torch.sparse_coo_tensor( s.indices(), block_seq, size=(*t_dim, *block_shape) ).coalesce()

def sparse_coo_2_hybrid_placeholder(s, block_shape, dtype, device):
    '''
    s is a sparse COO tensor. Any non-zero element in s indicates a block of size block_shape.
    This function returns a new sparse hybrid COO tensor, which has the same block structure as s.
    However, all the actual values of a block are zero.
    '''
    
    # Make sure we have ordered values.
    s = s.coalesce()
    
    # Only use the sparse dimension.
    t_dim = s.shape[:2]
    
    # Number of block and number of elements per block.
    n_block = s.values().shape[0]
    n_block_elem = block_shape[0] * block_shape[1]
    
    all_zero = torch.zeros(n_block * n_block_elem, dtype=dtype, device=device)
    all_zero = all_zero.view((n_block, *block_shape))
    
    return torch.sparse_coo_tensor( s.indices(), all_zero, size=(*t_dim, *block_shape) ).coalesce()

def hybrid_2_coo(hybrid):
    '''
    Covnert a sparse hybrid COO tensor to a sparse COO tensor.
    '''
    hybrid = hybrid.coalesce()
    indices_new, coo_dim = make_coo_indices_and_dims_from_hybrid(hybrid)
    return torch.sparse_coo_tensor(indices_new, hybrid.values().view((-1,)), size=coo_dim)

def coo_2_hybrid(coo, proxy):
    '''
    Convert a sparse COO tensor to a sparse hybrid COO tensor by referring to the proxy.
    
    A proxy is a sparse COO tensor. Any non-zero element in the proxy indicates a block.
    '''
    proxy = proxy.coalesce()
    
    # Figure out the shape of the target hybrid tensor.
    t_dim = proxy.shape[:2]
    assert coo.shape[0] % t_dim[0] == 0 and coo.shape[1] % t_dim[1] == 0, \
        f'coo and t_dim are not compatible: coo.shape = {coo.shape}, t_dim = {t_dim}. '
    b_dim = [ coo.shape[0] // t_dim[0], coo.shape[1] // t_dim[1] ]
    
    # Create a temporary sparse hybrid COO tensor to represent the block sequence.
    block_seq = sparse_coo_2_hybrid_block_sequence(proxy, b_dim)
    block_seq = hybrid_2_coo(block_seq).coalesce()
    
    # Create a temporary sparse hybrid COO tensor for the placeholders.
    block_phd = sparse_coo_2_hybrid_placeholder(proxy, b_dim, dtype=coo.dtype, device=coo.device)
    # PyTorch may has a bug here. If one of the tensors is coalesced, the the add operation will 
    # result in a coalesced tensor, no matter whether the other tensor is coalesced or not.
    # block_phd = hybrid_2_coo(block_phd).coalesce() 
    block_phd = hybrid_2_coo(block_phd)
    
    # Force the input coo tensor to have the same struture as block_seq and block_phd.
    coo = coo + block_phd
    coo = coo.coalesce()
    
    # Compute the indices of every element of coo inside their own respective blocks.
    b_dim_t = torch.Tensor([*b_dim]).to(dtype=torch.int64, device=coo.device).view((2, 1))
    in_block_indices = coo.indices() % b_dim_t
    
    # Index into a temporary tensor.
    n_block = proxy.values().shape[0]
    blocks = torch.zeros( (n_block, *b_dim), dtype=coo.dtype, device=coo.device )
    blocks[ block_seq.values(), in_block_indices[0], in_block_indices[1] ] = coo.values()

    # Create the sparse hybrid COO tensor.
    return torch.sparse_coo_tensor(proxy.indices(), blocks.view((n_block, *b_dim)), size=(*t_dim, *b_dim))

class SBTOperation(object):
    def __init__(self, func_name):
        super().__init__()
        self.func_name = func_name
        
    def storage_pre(self, func, types, args=(), kwargs={}):
        return args, args
    
    def storage_op(self, func, stripped_types, s_args=(), kwargs={}):
        return torch.Tensor.__torch_function__(func, stripped_types, s_args, kwargs)
    
    def proxy_op(self, func, stripped_types, p_args=(), kwargs={}):
        return p_args
    
    def storage_post(self, func, types, s_outs=(), p_outs=(), kwargs={}):
        return s_outs, p_outs

class SBTProxyNoOp(SBTOperation):
    def __init__(self, func_name):
        super().__init__(func_name=func_name)

    def storage_pre(self, func, types, args=(), kwargs={}):
        s_array = []
        p_array = []
        for arg in args:
            # TODO: Convert the sparse hybrid Tensor _s to sparse coo tensor.
            s_array.append( arg._s if isinstance(arg, SparseBlockTensor) else arg )
            
            # Do nothing about the proxy Tensor.
            p_array.append( arg._p if isinstance(arg, SparseBlockTensor) else arg )
        return s_array, p_array
    
    def proxy_op(self, func, stripped_types, p_args=(), kwargs={}):
        # Defaut operation on the proxy Tensor.
        # Assume that the operation does not need to even touch the
        # proxy tensor. For most of such operations, the proxy Tensor 
        # is the only sparse Tensor in the list of arguments.
        # Find the first sparse Tensor in operands.
        p = [ op for op in p_args 
                if isinstance(op, torch.Tensor) and 
                   op.is_sparse == True ][0]
        return p
    
    def storage_post(self, func, types, s_outs=(), p_outs=(), kwargs={}):
        # s_outs (outs for storage _s) and p_outs (outs for proxy _p) are
        # assumed to have the exact same order in the list.
        # Recover the block structure of s_outs.
        return s_outs, p_outs

class SBTProxySameOperationAsStorage(SBTOperation):
    def __init__(self, func_name):
        super().__init__(func_name)

    # NOTE: For test use.
    def storage_pre(self, func, types, args=(), kwargs={}):
        s_array = []
        p_array = []
        for arg in args:
            # Convert the sparse hybrid Tensor _s to sparse coo tensor.
            s_array.append( hybrid_2_coo( arg._s ) if isinstance(arg, SparseBlockTensor) else arg )
            
            # Do nothing about the proxy Tensor.
            p_array.append( arg._p if isinstance(arg, SparseBlockTensor) else arg )
        return s_array, p_array

    def proxy_op(self, func, stripped_types, p_args=(), kwargs={}):
        # This only gets called when the operation on sbt._s returns sparse Tensor.
        return torch.Tensor.__torch_function__(func, stripped_types, p_args, kwargs)
    
    def storage_post(self, func, types, s_outs=(), p_outs=(), kwargs={}):
        # s_outs (outs for storage _s) and p_outs (outs for proxy _p) are
        # assumed to have the exact same order in the list.
        # Recover the block structure of s_outs.
        
        h_outs = [ coo_2_hybrid(s, p) 
                    if isinstance(s, torch.Tensor) and s.is_sparse == True 
                    else s 
                    for s, p in zip(s_outs, p_outs) ]
        
        return h_outs, p_outs

_HANDLED_FUNCS_SPARSE = dict()

def _add_sparse_op(name, cls):
    global _HANDLED_FUNCS_SPARSE
    _HANDLED_FUNCS_SPARSE[name] = cls(name)

# Any supported operations that result in torch.Tensor should be added here.
_add_sparse_op( '__format__', SBTOperation )
_add_sparse_op( 'abs',    SBTProxyNoOp )
_add_sparse_op( 'matmul', SBTProxySameOperationAsStorage )

def _is_handled_func(func_name):
    return func_name in _HANDLED_FUNCS_SPARSE

# _HANDLED_FUNCS_SPARSE = [
#     SBTOperationMetaData('matmul', True), #'smm',

#     'is_sparse', 'dense_dim','sparse_dim', 'to_dense', 'values',
#     #'coalesce', 'is_coalesced', 'indices' COO only
#     #'crow_indices', 'col_indices' CSR and BSR only
# ] # decided according to "https://pytorch.org/docs/stable/sparse.html"


class SparseBlockTensor(torch.Tensor):

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs={}):
        # Debug use.
        print(f'func.__name__ = {func.__name__}')
        
        if not _is_handled_func(func.__name__):
            raise Exception(
                f'All operations on SparseBlockTensor must be handled. \n{func.__name__} is not. ')
        
        sbt_op = _HANDLED_FUNCS_SPARSE[func.__name__]
        
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
            
        outputs_list_storage, outputs_list_proxy = sbt_op.storage_post(func, stripped_types, outputs_storage, outputs_proxy, kwargs)

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
        r"""
        t = SparseBlockTensor()
        >>>t
        SparseBlockTensor()
        """
        return str(self)

    def __str__(self):
        r"""
        t = SparseBlockTensor()
        print( t )
        ' SparseBlockTensor() '
        """
        return f"SparseBlockTensor\nstorage:\n{self._s}\nproxy:\n{self._p}"

    def __format__(self, spec):
        return str(self)

    def matmul(self, other):
        r'''
        return the corresponding sparse matrix and index matrix
        '''
        print(f'>>> Debug matmtl. ')
        return self

    def __add__(self, other):
        pass

    def __mul__(self, other):
        pass


def sparse_block_tensor(indices, values, size=None, dtype=None, device=None, requires_grad=False):
    # Figure out the block shape.
    n_block, block_shape = values.shape[0], values.shape[1:]
    
    # Storage.
    storage = torch.sparse_coo_tensor(
        indices, 
        values, 
        size=(*size, *block_shape), 
        dtype=dtype, 
        device=device, 
        requires_grad=requires_grad ).coalesce()
    
    proxy = torch.sparse_coo_tensor(
        indices,
        torch.ones(n_block, dtype=dtype, device=device),
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
    
    i = torch.Tensor([
        [0, 0, 1, 1, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
        [0, 1, 0, 1, 4, 5, 4, 5, 2, 3, 2, 3, 4, 5, 4, 5] ]).to(dtype=torch.int64)
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
