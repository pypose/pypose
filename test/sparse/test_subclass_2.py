
import torch
from torch.utils._pytree import tree_map, tree_flatten

class SBTOperation(object):
    def __init__(self, func_name):
        super().__init__()
        self.func_name = func_name

    def storage_pre(self, func, types, args=(), kwargs={}):
        s_array = []
        p_array = []
        for arg in args:
            # Convert the sparse hybrid Tensor _s to sparse coo tensor.
            s_array.append( arg._s if isinstance(arg, SparseBlockTensor) else arg )
            
            p_array.append( arg._p if isinstance(arg, SparseBlockTensor) else arg )
        return s_array, p_array

    def storage_post(self, func, types, s_outs=(), p_outs=(), kwargs={}):
        # s_outs (outs for storage _s) and p_outs (outs for proxy _p) are
        # assumed to have the exact same order in the list.
        return s_outs, p_outs
    
    def proxy_op(self, func, types, p_args=(), kwargs={}):
        # Defaut operation on the proxy Tensor.
        # Assume that the operation does not need to even touch the
        # proxy tensor. For most of such operations, the proxy Tensor 
        # is the only sparse Tensor in the list of arguments.
        # Find the first sparse Tensor in operands.
        p = [ op for op in p_args 
                if isinstance(op, torch.Tensor) and 
                   op.is_sparse == True ][0]
        return p

class SBTProxySameOperationAsStorage(SBTOperation):
    def __init__(self, func_name):
        super().__init__(func_name)

    def proxy_op(self, func, types, p_args=(), kwargs={}):
        # This only gets called when the operation on sbt._s returns sparse Tensor.
        torch.Tensor.__torch_function__(func, types, p_args, kwargs)

_HANDLED_FUNCS_SPARSE = dict()

def _add_sparse_op(name, cls):
    global _HANDLED_FUNCS_SPARSE
    _HANDLED_FUNCS_SPARSE[name] = cls(name)

# Any supported operations that result in torch.Tensor should be added here.
_add_sparse_op( 'abs',    SBTProxyOperation )
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
        global _HANDLED_FUNCS_SPARSE, _FUNCS_UPDATE_PROXY

        mytypes = (torch.Tensor if t is SparseBlockTensor else t for t in types)
        myargs = ( t.sbt if isinstance(t, SparseBlockTensor) else t for t in args)
        myproxies = ( t.dummy if isinstance(t, SparseBlockTensor) else t for t in args)
        print(f'func.__name__ = {func.__name__}')
        # t.sbt is the sparse matrix
        # t.dummy is the corresponding proxy matrix
        data = torch.Tensor.__torch_function__(func, mytypes, myargs, kwargs)

        if func.__name__ in _FUNCS_UPDATE_PROXY:
            data_p = torch.Tensor.__torch_function__(func, mytypes, myproxies, kwargs)
        else:
            data_p

        # Recover the types.
        if data is not None and func.__name__ in _HANDLED_FUNCS_SPARSE:
            args, spec = tree_flatten(args)
            def wrap(t, p):
                if isinstance(t, torch.Tensor) and not isinstance(t, cls):
                    if not t.is_sparse:
                        return t
                    
                    # Wrap t and p into a SparseBlockTensor.
                    s = cls()
                    s.sbt = t
                    s.dummy = p
                    return s 

                return t
            return tree_map(warp, )

                



        # if func.__name__ in _HANDLED_FUNCS_SPARSE:
        #     print("enter func.__name__ : line 19 ")
        #     #out = MyTensor()
        #     #out.sbt = data[0]
        #     return out


        # else:
        #     out = data

        return data

    def __repr__(self):
        r"""
        t = SparseBlockTensor()
        >>>t
        SparseBlockTensor()
        """
        return "SparseBlockTensor()"

    def __str__(self):
        r"""
        t = SparseBlockTensor()
        print( t )
        ' SparseBlockTensor() '
        """
        return "SparseBlockTensor"


    def __matmul__(self, other):
        r'''
        return the corresponding sparse matrix and index matrix
        '''
        #print("in here")

        out = SparseBlockTensor()
        out.sbt = self.sbt @ other.sbt
        out.dummy = self.dummy @ other.dummy # Need to change to the represent matrix ( COO )user specified !!!!!!!!! 
        #print(f"out.sbt = {out.sbt} ")
        #print(f"out.dummy = {out.dummy}" )
        return out

    def __add__(self, other):
        pass

    def __mul__(self, other):
        pass


def sparse_block_tensor(indices, values, size=None, dtype=None, device=None, requires_grad=False):
    data = torch.sparse_coo_tensor(indices, values, size=size, dtype=dtype, device=device, requires_grad=requires_grad)
    x = SparseBlockTensor()
    x._s = data # s for storage.
    x._p = data # for now need to modify
    return x

def test_simple():
    print()

    i = [[0, 1, 2],[2, 0, 2]]
    v = [3, 4, 5]
    x = sparse_block_tensor(i, v, size=(3, 3), dtype=torch.float32)

    print(f'type(x) = {type(x)}')
    print(f'x._s = {x._s}')
    
    #print(x)

    #y = x.to_dense()
    #print(y)

    z = x @ x
    #print(z)
    #print(f'z = {z}')
    #print(f'type(z) = {type(z)}')

if __name__ == '__main__':
    test_simple()
