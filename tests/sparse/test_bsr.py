import torch
import pytest
import pypose as pp

def random_compressed(proxy_shape, block_shape, shape_mode, dense_zero_prob=0.):
    #generate coo
    proxy = torch.randn(proxy_shape) > 0.5
    coo_indices = proxy.nonzero().T  # (dim, nnz)
    values = torch.randn((coo_indices.shape[1], *block_shape))
    values[torch.rand(values.shape) < dense_zero_prob] = 0
    m = proxy_shape[-2] * block_shape[-2]
    p = proxy_shape[-1] * block_shape[-1]

    dummy_val = torch.zeros(coo_indices.shape[-1], dtype=values.dtype, device=values.device)
    dummy = torch.sparse_coo_tensor(indices=coo_indices,
                                        values=dummy_val,
                                        size=(m, p)).coalesce()

    if shape_mode == 'bsr' :
        dummy_csr = dummy.to_sparse_csr()
        return torch.sparse_bsr_tensor(dummy_csr.crow_indices(),
                                    dummy_csr.col_indices(),
                                    values,
                                    size=(m, p), dtype=values.dtype)
    elif shape_mode == 'bsc' :
        dummy_csc = dummy.to_sparse_csc()
        return torch.sparse_bsc_tensor(dummy_csc.ccol_indices(),
                                    dummy_csc.row_indices(),
                                    values,
                                    size=(m, p), dtype=values.dtype)

@pytest.mark.parametrize('dense_zero_prob', [0., 0.7, 1.0])
@pytest.mark.parametrize('op, dense_op, type_operands, shape_mode, dim', [
    (torch.matmul, torch.matmul, ['bsr', 'bsc'], 'mT', 2),
    (torch.matmul, torch.matmul, ['bsr', 'bsc'], 'identical_square', 2)])
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
        proxy_shapes = [proxy_shape if idx % 2 == 0 else shape_mT(proxy_shape)\
                            for idx, _ in enumerate(type_operands)]
        block_shapes = [block_shape if idx % 2 == 0 else shape_mT(block_shape)\
                            for idx, _ in enumerate(type_operands)]
    else:
        raise ValueError(f'Unknown shape_mode: {shape_mode}')
    dense_shapes = [torch.Size(torch.tensor(proxy_shape) * torch.tensor(block_shape))
                    for proxy_shape, block_shape in zip(proxy_shapes, block_shapes)]
    args = []
    args_dense = []
    for t, proxy_shape, block_shape, dense_shape in zip(type_operands, proxy_shapes,
                                                        block_shapes, dense_shapes):
        arg = random_compressed(proxy_shape, block_shape, t, dense_zero_prob)
        arg_dense = arg.to_dense()
        assert arg_dense.shape == dense_shape
        args.append(arg)
        args_dense.append(arg_dense)
    y_sbt = op(*args)
    y_dense = dense_op(*args_dense)

    torch.testing.assert_close(y_sbt.to_sparse_coo().to_dense(), y_dense, equal_nan=True)
