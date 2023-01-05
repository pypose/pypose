
import pytest

import torch

_HANDLED_FUNCS_SPARSE = [
    'matmul'
]
class MyTensor(torch.Tensor):

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs={}):
        global _HANDLED_FUNCS_SPARSE

        mytypes = (torch.Tensor if t is MyTensor else t for t in types)
        myargs = (t.sbt if isinstance(t, MyTensor) else t for t in args)
        res = torch.Tensor.__torch_function__(func, mytypes, myargs, kwargs)

        print(f'func.__name__ = {func.__name__}')
        if func.__name__ in _HANDLED_FUNCS_SPARSE:
            out = MyTensor()
            out.sbt = res
        else:
            out = res

        return out


def sparse_block_tensor(indices, values, size=None, dtype=None, device=None, requires_grad=False):
    data = torch.sparse_coo_tensor(indices, values, size=size, dtype=dtype, device=device, requires_grad=requires_grad)
    x = MyTensor()
    x.sbt = data
    return x

def test_simple():
    print()

    i = [[0, 1, 2],[2, 0, 2]]
    v = [3, 4, 5]
    x = sparse_block_tensor(i, v, size=(3, 3), dtype=torch.float32)

    print(f'type(x) = {type(x)}')
    print(f'x.sbt = {x.sbt}')
    
    print(x)

    y = x.to_dense()
    print(y)

    z = x @ x
    print(f'type(z) = {type(z)}')