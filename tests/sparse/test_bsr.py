import torch
import pytest
import pypose as pp

def random_compressed(pshape, bshape, mode, zero_prob=0.):
    #generate coo
    proxy = torch.randn(pshape) > 0.5
    coo_indices = proxy.nonzero().T  # (dim, nnz)
    values = torch.randn((coo_indices.shape[-1], *bshape))
    values[torch.rand(values.shape) < zero_prob] = 0
    m = pshape[-2] * bshape[-2]
    p = pshape[-1] * bshape[-1]

    dummy_val = torch.zeros(coo_indices.shape[-1], dtype=values.dtype)
    dummy = torch.sparse_coo_tensor(coo_indices, dummy_val, size=(m, p)).coalesce()

    if mode == 'bsr' :
        dummy_csr = dummy.to_sparse_csr()
        crowi, coli = dummy_csr.crow_indices(), dummy_csr.col_indices()
        return torch.sparse_bsr_tensor(crowi, coli, values, (m, p), dtype=values.dtype)

    elif mode == 'bsc' :
        dummy_csc = dummy.to_sparse_csc()
        crowi, coli = dummy_csc.ccol_indices(), dummy_csc.row_indices()
        return torch.sparse_bsc_tensor(crowi, coli, values, (m, p), dtype=values.dtype)


class TestBSR:
    @pytest.mark.parametrize('zero_prob', [0., 0.7, 1.0])
    @pytest.mark.parametrize('op, dense_op, layouts, mode, dim', [
        (torch.matmul, torch.matmul, ['bsr', 'bsc'], 'mT', 2),
        (torch.matmul, torch.matmul, ['bsr', 'bsc'], 'identical_square', 2)])
    def test_universal(self, op, dense_op, layouts, mode, dim, zero_prob):
        if mode == 'identical':
            pshape = torch.Size(torch.randint(1, 10, (dim,)))
            bshape = torch.Size(torch.randint(1, 10, (dim,)))
            pshapes = [pshape for _ in layouts]
            bshapes = [bshape for _ in layouts]
        elif mode == 'identical_square':
            pshape = torch.Size(torch.randint(1, 10, (dim - 1,)))
            pshape = pshape + pshape[-1:]
            bshape = torch.Size(torch.randint(1, 10, (dim - 1,)))
            bshape = bshape + bshape[-1:]
            pshapes = [pshape] * len(layouts)
            bshapes = [bshape] * len(layouts)
        elif mode == 'mT':
            shape_mT = lambda shape: shape[:-2] + torch.Size([shape[-1], shape[-2]])
            pshape = torch.Size(torch.randint(1, 10, (dim,)))
            bshape = torch.Size(torch.randint(1, 10, (dim,)))
            pshapes = [pshape if idx % 2 == 0 else shape_mT(pshape)\
                                for idx, _ in enumerate(layouts)]
            bshapes = [bshape if idx % 2 == 0 else shape_mT(bshape)\
                                for idx, _ in enumerate(layouts)]
        else:
            raise ValueError(f'Unknown shape mode: {mode}')
        dshapes = [torch.Size(torch.tensor(pshape) * torch.tensor(bshape))
                        for pshape, bshape in zip(pshapes, bshapes)]
        args, args_dense = [], []
        for t, pshape, bshape, dshape in zip(layouts, pshapes, bshapes, dshapes):
            arg = random_compressed(pshape, bshape, t, zero_prob)
            arg_dense = arg.to_dense()
            assert arg_dense.shape == dshape
            args.append(arg)
            args_dense.append(arg_dense)
        y_sparse = op(*args)
        y_dense = dense_op(*args_dense)

        torch.testing.assert_close(y_sparse.to_dense(), y_dense, equal_nan=True)


if __name__ == '__main__':
    TestBSR.test_universal(None, torch.matmul, torch.matmul, ['bsr', 'bsc'], 'mT', 2, 0.7)

    crow_indices = torch.tensor([0, 2, 4])
    col_indices = torch.tensor([0, 1, 0, 1])
    values = torch.tensor([[[0, 1, 2], [6, 7, 8]],
                           [[3, 4, 5], [9, 10, 11]],
                           [[12, 13, 14], [18, 19, 20]],
                           [[15, 16, 17], [21, 22, 23]]])
    bsr = torch.sparse_bsr_tensor(crow_indices, col_indices, values, dtype=torch.float64)
    import time
    start = time.perf_counter()
    for _ in range(1000):
        bsr @ bsr.mT
    end = time.perf_counter()
    print(end - start)
