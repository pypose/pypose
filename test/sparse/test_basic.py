
import torch
import pypose as pp
from pypose.sparse import sparse_block_tensor, coo_2_hybrid, hybrid_2_coo
import pypose.sparse as sp

def test_pypose_operation():
    a = pp.randn_so3(2, requires_grad=True)
    b = pp.randn_so3(2, requires_grad=True)
    c = a + b
    print(f'c = \n{c}')
    d = torch.add(a, b)
    print(f'd = \n{d}')
    e = pp.add(a, b)
    print(f'e = \n{e}')

def test_sparse_coo_2_sparse_hybrid_coo():
    print()

    i = [[0, 1, 2],[2, 0, 2]]
    v = torch.Tensor([[3, 4], [-5, -6], [7, 8]]).to(dtype=torch.float32)
    x = sparse_block_tensor(i, v, size=(3, 3), dtype=torch.float32)

    print(f'type(x) = {type(x)}')
    print(f'x._s = \n{x._s}')
    print(f'x._p = \n{x._p}')

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

    y2 = torch.abs(x)
    print(f'y2 = \n{y2}')

    y3 = sp.abs(x)
    print(f'y3 = \n{y3}')
    
    #print(x)

    #y = x.to_dense()
    #print(y)

    # z = x @ x
    #print(z)
    #print(f'z = {z}')
    #print(f'type(z) = {type(z)}')

def test_mm():
    print()

    i = [[0, 0, 1, 2],[0, 2, 1, 2]]
    v = torch.arange(16).view((-1, 2, 2)).to(dtype=torch.float32)
    x = sparse_block_tensor(i, v, size=(3, 3), dtype=torch.float32)

    # Show the storage and proxy.
    print(f'type(x) = {type(x)}')
    print(f'x._s = \n{x._s}')
    print(f'x._p = \n{x._p}')
    
    # Perform matrix multiplication using the proxies.
    m = x._p @ x._p
    print(f'm = \n{m}')
    
    # Perform matrix multiplication.
    y = x @ x
    print(f'y = \n{y}')
    
    # Compute the true multiplication result.
    xh = hybrid_2_coo(x._s)
    print(f'xh = \n{xh.to_dense()}')
    yh0 = xh @ xh
    
    # Convert our result.
    yh = hybrid_2_coo(y._s)
    
    # Show and compare.
    print(f'yh0 = \n{yh0.to_dense()}')
    print(f'yh = \n{yh.to_dense()}')

def test_is_sparse():
    print()

    i = [[0, 0, 1, 2],[0, 2, 1, 2]]
    v = torch.arange(16).view((-1, 2, 2)).to(dtype=torch.float32)
    x = sparse_block_tensor(i, v, size=(3, 3), dtype=torch.float32)

    iss = x.is_sparse
    print(f'x.is_sparse = {iss}')
