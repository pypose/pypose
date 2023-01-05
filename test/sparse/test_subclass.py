
import torch

_HANDLED_FUNCS_SPARSE = [
    'matmul','smm',
    'is_sparse', 'dense_dim','sparse_dim', 'to_dense', 'values',
    #'coalesce', 'is_coalesced', 'indices' COO only
    #'crow_indices', 'col_indices' CSR and BSR only
] # decided according to "https://pytorch.org/docs/stable/sparse.html"

class SparseBlockTensor(torch.Tensor):

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs={}):
        global _HANDLED_FUNCS_SPARSE

        mytypes = (torch.Tensor if t is SparseBlockTensor else t for t in types)
        myargs = ( (t.sbt, t.dummy) if isinstance(t, SparseBlockTensor) else t for t in args)
        # t.sbt is the sparse matrix
        # t.dummy is the corresponding represent matrix
        data = torch.Tensor.__torch_function__(func, mytypes, myargs, kwargs)

        print(f'func.__name__ = {func.__name__}')

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
    x.sbt = data
    x.dummy = data # for now need to modify
    return x

def test_simple():
    print()

    i = [[0, 1, 2],[2, 0, 2]]
    v = [3, 4, 5]
    x = sparse_block_tensor(i, v, size=(3, 3), dtype=torch.float32)

    print(f'type(x) = {type(x)}')
    print(f'x.sbt = {x.sbt}')
    
    #print(x)

    #y = x.to_dense()
    #print(y)

    z = x @ x
    #print(z)
    #print(f'z = {z}')
    #print(f'type(z) = {type(z)}')

if __name__ == '__main__':
    test_simple()
