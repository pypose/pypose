
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
    i = [[0, 1, 2],[2, 0, 2]]
    v = torch.tensor([[3, 4], [-5, -6], [7, 8]], dtype=torch.float32)
    x = sparse_block_tensor(i, v, size=(3, 3), dtype=torch.float32)

    print(f'type(x) = {type(x)}')
    print(f'x._s = \n{x._s}')
    print(f'x._p = \n{x._p}')


def random_sbt(proxy_shape, block_shape):
    proxy = torch.randn(proxy_shape) > 0.5
    indices = proxy.nonzero().T  # (dim, nnz)
    values = torch.randn((indices.shape[1], *block_shape))
    return sparse_block_tensor(indices, values, size=proxy_shape)


@pytest.mark.parametrize('meta', [
    (sp.abs, torch.abs, 1, 'identical'),
    (torch.abs, torch.abs, 1, 'identical'),
    (SparseBlockTensor.__add__, torch.add, 2, 'identical'),
    (torch.add, torch.add, 2, 'identical'),
    (SparseBlockTensor.__sub__, torch.sub, 2, 'identical'),
    (SparseBlockTensor.__matmul__, torch.matmul, 2, 'identical_square'),],
    )
def test_universal(meta, dim=2):
    op, dense_op, num_operands, shape_mode = meta

    if shape_mode == 'identical':
        proxy_shape = torch.Size(torch.randint(1, 10, (dim,)))
        block_shape = torch.randint(1, 10, (dim,))
        proxy_shapes = [proxy_shape for _ in range(num_operands)]
        block_shapes = [block_shape for _ in range(num_operands)]
    elif shape_mode == 'identical_square':
        proxy_shape = torch.Size(torch.randint(1, 10, (1,))) * 2
        block_shape = torch.Size(torch.randint(1, 10, (1,))) * 2
        proxy_shapes = [proxy_shape for _ in range(num_operands)]
        block_shapes = [block_shape for _ in range(num_operands)]
    else:
        raise ValueError(f'Unknown shape_mode: {shape_mode}')
    args = [random_sbt(proxy_shape, block_shape) for _ in zip(proxy_shapes, block_shapes)]
    y_sbt = op(*args)

    # dense reference
    args_dense = [hybrid_2_coo(x._s).to_dense() for x in args]
    y_dense = dense_op(*args_dense)

    torch.testing.assert_close(hybrid_2_coo(y_sbt._s).to_dense(), y_dense)


def test_mm():
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

    assert x.is_sparse


if __name__ == '__main__':
    i = [[0, 0, 1, 2],[0, 2, 1, 2]]
    v = torch.arange(16).view((4, 2, 2)).to(dtype=torch.float32)
    x = sparse_block_tensor(i, v, size=(3, 3), dtype=torch.float32)
    x = hybrid_2_coo(x._s)

    import time
    start = time.time()
    for i in range(10000):
        x @ x
    end = time.time()
    print(f'elapsed time = {end - start}')

    from tqdm import tqdm
    mm_config = (SparseBlockTensor.__matmul__, torch.matmul, 2, 'identical_square')
    for i in tqdm(range(10000)):
        test_universal(mm_config)
