import torch
import pytest
import pypose as pp
from pypose.sparse import bsr_mm_triton, bsr_output_to_dense_numpy

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

def test_triton_bsr_mm():
    for run_idx in range(20):
        pshape = torch.Size([4, 4])
        bshape = torch.Size([2, 2])
        A_bsr = random_compressed(pshape, bshape, 'bsr', zero_prob=0.3)
        B_bsr = random_compressed(pshape, bshape, 'bsr', zero_prob=0.3)
        A_dense = A_bsr.to_dense().to(torch.float32).cuda()
        B_dense = B_bsr.to_dense().to(torch.float32).cuda()
        C_dense_reference = A_dense @ B_dense
        A_offsets = A_bsr.crow_indices().to(torch.int32).cuda()
        A_cols = A_bsr.col_indices().to(torch.int32).cuda()
        A_vals = A_bsr.values().to(torch.float32).cuda()
        B_offsets = B_bsr.crow_indices().to(torch.int32).cuda()
        B_cols = B_bsr.col_indices().to(torch.int32).cuda()
        B_vals = B_bsr.values().to(torch.float32).cuda()
        A_block_rows = A_vals.shape[1]
        A_block_cols = A_vals.shape[2]
        B_block_cols = B_vals.shape[2]

        C_offsets, C_cols, C_vals = bsr_mm_triton(
            A_offsets, A_cols, A_vals,
            B_offsets, B_cols, B_vals,
            A_bsr.shape[0] // A_block_rows,
            A_bsr.shape[1] // A_block_cols,
            B_bsr.shape[1] // B_block_cols,
        )

        C_dense_triton = bsr_output_to_dense_numpy(
            C_offsets.cpu().numpy(),
            C_cols.cpu().numpy(),
            C_vals.cpu().numpy(),
            A_bsr.shape[0] // A_block_rows,
            B_bsr.shape[1] // B_block_cols,
            A_block_rows,
            B_block_cols,
        )

        C_dense_triton = torch.from_numpy(C_dense_triton).cuda()
        torch.testing.assert_close(C_dense_triton, C_dense_reference, rtol=1e-4, atol=1e-4)
        print(f"Test for {run_idx+1} of 20 passed ")

    print("All 20 Triton BSR Muiltiplication tests passed")


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
    test_triton_bsr_mm()
