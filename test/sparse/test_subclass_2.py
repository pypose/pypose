
import torch
from torch.utils._pytree import tree_map, tree_flatten

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

    def proxy_op(self, func, stripped_types, p_args=(), kwargs={}):
        # This only gets called when the operation on sbt._s returns sparse Tensor.
        return torch.Tensor.__torch_function__(func, stripped_types, p_args, kwargs)

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
        if func.__name__ == 'matmul':
            import ipdb; ipdb.set_trace()
            pass
        
        # Let PyTorch do its dispatching.
        outputs_storage = sbt_op.storage_op(func, stripped_types, args_storage, kwargs)

        # Handle the proxy.
        outputs_proxy = sbt_op.proxy_op(func, stripped_types, args_proxy, kwargs)
        
        # Do post processing.
        outputs_storage, outputs_proxy = sbt_op.storage_post(func, stripped_types, outputs_storage, outputs_proxy, kwargs)

        
        if outputs_storage is None:
            return None

        # Recover the types.
        if isinstance(outputs_storage, torch.Tensor) and not isinstance(outputs_storage, cls):
            if not outputs_storage.is_sparse:
                return outputs_storage
            
            # Wrap s and p into a SparseBlockTensor.
            sbt = cls()
            sbt._s = outputs_storage
            sbt._p = outputs_proxy
            return sbt
        # Not a tensor or not a SparseBlockTensor.
        return outputs_storage

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
        requires_grad=requires_grad )
    
    proxy = torch.sparse_coo_tensor(
        indices,
        torch.ones(n_block, dtype=dtype, device=device),
        size=size,
        dtype=dtype,
        device=device,
        requires_grad=False )
    
    x = SparseBlockTensor()
    x._s = storage # s for storage.
    x._p = proxy
    return x

def test_abs():
    print()

    i = [[0, 1, 2],[2, 0, 2]]
    v = torch.Tensor([[3, 4], [-5, -6], [7, 8]]).to(dtype=torch.float32)
    x = sparse_block_tensor(i, v, size=(3, 3), dtype=torch.float32)

    print(f'type(x) = {type(x)}')
    print(f'x._s = \n{x._s}')
    print(f'x._p = \n{x._p}')
    
    y = x.abs()
    print(f'y = \n{y}')
    
    #print(x)

    #y = x.to_dense()
    #print(y)

    # z = x @ x
    #print(z)
    #print(f'z = {z}')
    #print(f'type(z) = {type(z)}')

def test_mm():
    print()

    i = [[0, 0, 1, 2],[0, 1, 1, 2]]
    v = torch.arange(16).view((-1, 2, 2)).to(dtype=torch.float32)
    x = sparse_block_tensor(i, v, size=(3, 3), dtype=torch.float32)

    print(f'type(x) = {type(x)}')
    print(f'x._s = \n{x._s}')
    print(f'x._p = \n{x._p}')
    
    m = x._s @ x._s
    print(f'm = \n{m}')
    
    y = x @ x
    print(f'y = \n{y}')

if __name__ == '__main__':
    test_abs()
    test_mm()