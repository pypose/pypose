import numpy as np
import pytest
import torch
import pypose as pp
from pypose.sparse import sparse_block_tensor, coo_2_hybrid, hybrid_2_coo, SparseBlockTensor
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
    i = [[0, 1, 2], [2, 0, 2]]
    v = torch.tensor([[3, 4], [-5, -6], [7, 8]], dtype=torch.float32)
    x = sparse_block_tensor(i, v, size=(3, 3), dtype=torch.float32)

    print(f'type(x) = {type(x)}')
    print(f'x._s = \n{x._s}')
    print(f'x._p = \n{x._p}')


def random_sbt(proxy_shape, block_shape, dense_zero_prob=0.):
    proxy = torch.randn(proxy_shape) > 0.5
    indices = proxy.nonzero().T  # (dim, nnz)
    values = torch.randn((indices.shape[1], *block_shape))
    values[torch.rand(values.shape) < dense_zero_prob] = 0
    return sparse_block_tensor(indices, values, size=proxy_shape)


@pytest.mark.parametrize('dense_zero_prob', [0., 0.7, 1.0])
@pytest.mark.parametrize('op,dense_op,type_operands,shape_mode,dim', [
    (sp.abs, torch.abs, ['sbt'], 'identical', 2),
    (sp.abs, torch.abs, ['sbt'], 'identical', 3),
    (torch.abs, torch.abs, ['sbt'], 'identical', 2),
    (torch.abs, torch.abs, ['sbt'], 'identical', 3),
    (SparseBlockTensor.ceil, torch.ceil, ['sbt'], 'identical', 2),
    (SparseBlockTensor.ceil, torch.ceil, ['sbt'], 'identical', 3),
    (SparseBlockTensor.asin, torch.asin, ['sbt'], 'identical', 2),
    (SparseBlockTensor.atan, torch.atan, ['sbt'], 'identical', 3),
    (SparseBlockTensor.sin, torch.sin, ['sbt'], 'identical', 2),
    (SparseBlockTensor.tan, torch.tan, ['sbt'], 'identical', 3),
    (SparseBlockTensor.sinh, torch.sinh, ['sbt'], 'identical', 2),
    (SparseBlockTensor.tanh, torch.tanh, ['sbt'], 'identical', 3),
    (SparseBlockTensor.floor, torch.floor, ['sbt'], 'identical', 2),
    (SparseBlockTensor.floor, torch.floor, ['sbt'], 'identical', 3),
    (SparseBlockTensor.round, torch.round, ['sbt'], 'identical', 2),
    (SparseBlockTensor.round, torch.round, ['sbt'], 'identical', 3),
    (SparseBlockTensor.sqrt, torch.sqrt, ['sbt'], 'identical', 2),
    (SparseBlockTensor.sqrt, torch.sqrt, ['sbt'], 'identical', 3),
    (SparseBlockTensor.square, torch.square, ['sbt'], 'identical', 2),
    (SparseBlockTensor.square, torch.square, ['sbt'], 'identical', 3),
    (SparseBlockTensor.__add__, torch.add, ['sbt', 'sbt'], 'identical', 2),
    (SparseBlockTensor.__add__, torch.add, ['sbt', 'sbt'], 'identical', 3),
    (SparseBlockTensor.__sub__, torch.sub, ['sbt', 'sbt'], 'identical', 2),
    (SparseBlockTensor.__sub__, torch.sub, ['sbt', 'sbt'], 'identical', 3),
    (SparseBlockTensor.__matmul__, torch.matmul, ['sbt', 'sbt'], 'mT', 2),
    (SparseBlockTensor.__matmul__, torch.matmul, ['sbt', 'sbt'], 'identical_square', 2), ],
                         )
def test_universal(op, dense_op, type_operands, shape_mode, dim, dense_zero_prob):
    if shape_mode == 'identical':
        proxy_shape = torch.Size(torch.randint(1, 10, (dim,)))
        block_shape = torch.Size(torch.randint(1, 10, (dim,)))
        proxy_shapes = [proxy_shape for _ in type_operands]
        block_shapes = [block_shape for _ in type_operands]
    elif shape_mode == 'identical_square':
        proxy_shape = torch.Size(torch.randint(1, 10, (dim - 1,)))
        proxy_shape = proxy_shape + proxy_shape[-1:]
        block_shape = torch.Size(torch.randint(1, 10, (dim - 1,)))
        block_shape = block_shape + block_shape[-1:]
        proxy_shapes = [proxy_shape for _ in type_operands]
        block_shapes = [block_shape for _ in type_operands]
    elif shape_mode == 'mT':
        shape_mT = lambda shape: shape[:-2] + torch.Size([shape[-1], shape[-2]])
        proxy_shape = torch.Size(torch.randint(1, 10, (dim,)))
        block_shape = torch.Size(torch.randint(1, 10, (dim,)))
        proxy_shapes = [proxy_shape if idx % 2 == 0 else shape_mT(proxy_shape) for idx, _ in enumerate(type_operands)]
        block_shapes = [block_shape if idx % 2 == 0 else shape_mT(block_shape) for idx, _ in enumerate(type_operands)]
    else:
        raise ValueError(f'Unknown shape_mode: {shape_mode}')
    dense_shapes = [torch.Size(np.multiply(proxy_shape, block_shape))
                    for proxy_shape, block_shape in zip(proxy_shapes, block_shapes)]
    args = []
    args_dense = []
    for t, proxy_shape, block_shape, dense_shapes in zip(type_operands, proxy_shapes, block_shapes, dense_shapes):
        if t == 'sbt':
            arg = random_sbt(proxy_shape, block_shape, dense_zero_prob)
            arg_dense = hybrid_2_coo(arg._s).to_dense()
            assert arg_dense.shape == dense_shapes
        elif t == 'dense':
            arg = torch.randn(dense_shapes)
            arg_dense = arg
        else:
            raise ValueError(f'Unknown type_operands: {t}')
        args.append(arg)
        args_dense.append(arg_dense)
    y_sbt = op(*args)
    y_dense = dense_op(*args_dense)

    torch.testing.assert_close(hybrid_2_coo(y_sbt._s).to_dense(), y_dense,equal_nan=True)

def test_div_beh():
    i = [[0, 0],
         [0, 1]]
    v = [[5.7, 5.1],
         [5.5, 5.0]]
    s = torch.sparse_coo_tensor(i, v, (1, 2, 2))

    i2 = [[0],
          [0]]
    v2 = [[0.1, -0.1]]
    s2 = torch.sparse_coo_tensor(i2, v2, (1, 2, 2))
    s*s2

    's.div(s2)'

def test_mm():
    i = [[0, 0, 1, 2], [0, 2, 1, 2]]
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
    i = [[0, 0, 1, 2], [0, 2, 1, 2]]
    v = torch.arange(16).view((-1, 2, 2)).to(dtype=torch.float32)
    x = sparse_block_tensor(i, v, size=(3, 3), dtype=torch.float32)

    assert x.is_sparse


if __name__ == '__main__':
    i = [[0, 0, 1, 2], [0, 2, 1, 2]]
    v = torch.arange(16).view((4, 2, 2)).to(dtype=torch.float32)
    x = sparse_block_tensor(i, v, size=(3, 3), dtype=torch.float32)
    # if testing dense / sparse, set the following
    # x = hybrid_2_coo(x._s)

    import time

    start = time.time()
    for i in range(10000):
        x @ x
    end = time.time()
    print(f'elapsed time = {end - start}')

    from tqdm import tqdm

    mm_config = (SparseBlockTensor.__matmul__, torch.matmul, 2, 'identical_square', 0.7)
    for i in tqdm(range(10000)):
        test_universal(*mm_config)
