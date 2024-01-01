import torch
import warnings
from typing import Callable, List, Optional
from torch.library import Library


def diagonal_op_(input, offset: int=0, op: Optional[Callable]=None):
    crow_indices = input.crow_indices() # b + 1 dimensional
    col_indices = input.col_indices() # b + 1 dimensional
    bsr_values = input.values() # 1 + 2 dimensional
    m, n = input.shape[-2], input.shape[-1]
    dm, dn = (bsr_values.shape[-2], bsr_values.shape[-1])
    sm, sn = m // dm, n // dn

    #simple case(block is square and offset is 0)
    if dm == dn and offset == 0:
        dummy_val = torch.zeros(bsr_values.shape[0], device='cpu')
        dummy = torch.sparse_csr_tensor(crow_indices=crow_indices.to('cpu'),
                                        col_indices=col_indices.to('cpu'),
                                        values=dummy_val)
        dummy_coo = dummy.to_sparse(layout=torch.sparse_coo).coalesce()

        indices = dummy_coo.indices().to(input.device)
        diag_indices = (indices[0] == indices[1]).nonzero().squeeze(-1)
        block_diags = bsr_values.diagonal(dim1=-2, dim2=-1)
        values = block_diags[diag_indices]
        n_diag_blocks = sm if sm < sn else sn
        if diag_indices.shape[-1] == n_diag_blocks:
            results = values
        else:
            results_shape = (n_diag_blocks, dm)
            results = torch.zeros(results_shape, dtype=values.dtype, device=values.device)
            results[indices[0, diag_indices]] = values
            assert op is None, "op is not supported for diagonal that has empty values."
        results = torch.flatten(results, start_dim=-2, end_dim=-1)
        # apply the inplace op
        if op is not None:
            results = op(results)
            block_diags[diag_indices] = results.view(n_diag_blocks, dm)
        return results
    else:
        raise NotImplementedError('Only square block and offset 0 is supported.')


def _sparse_csr_mm(mat1, mat2):
    if isinstance(mat1, torch.Tensor) and mat1.layout == torch.sparse_bsr:
        if isinstance(mat2, torch.Tensor) and mat2.layout == torch.sparse_bsc:
            return bsr_bsc_matmul(mat1, mat2)
    elif isinstance(mat1, torch.Tensor) and mat1.layout == torch.sparse_bsc:
        if isinstance(mat2, torch.Tensor) and mat2.layout == torch.sparse_bsr:
            raise NotImplemented
    #https://github.com/pytorch/pytorch/blob/3fa3ed4923c19a2b8d2da69e994169b4c8ac5fe3/
    #aten/src/ATen/native/sparse/SparseCsrTensorMath.cpp#L789
    if mat1.is_sparse_csr and mat2.is_sparse_csr:
        size = [mat1.size(0), mat2.size(1)]
        zero = torch.zeros(size, dtype=mat2.dtype, device=mat2.device, layout=mat2.layout)
        return torch.addmm(zero, mat1, mat2, beta=0.0, alpha=1.0)

    if (mat1.layout == torch.sparse_csc or mat1.layout == torch.sparse_csr) and\
        (mat2.layout == torch.sparse_csc or mat2.layout == torch.sparse_csr):
        return _sparse_csr_mm(mat1.to_sparse_csr(), mat2.to_sparse_csr())

    if mat1.layout == torch.sparse_csc and mat2.layout == torch.strided:
        return _sparse_csr_mm(mat1.to_sparse_csr(), mat2)

    if mat2.layout == torch.strided:
        size = [mat1.size(0), mat2.size(1)]
        zero = torch.zeros(size, dtype=mat1.dtype, device=mat1.device, layout=mat2.layout)
        return torch.addmm(zero, mat1, mat2, beta=0.0, alpha=1.0)

    size = [mat1.size(0), mat2.size(1)]
    zero = torch.zeros(size, dtype=mat1.dtype, device=mat1.device, layout=mat1.layout),
    return torch.addmm(zero, mat1, mat2, beta=0.0, alpha=1.0)


@torch.jit.script
def bsr_bsc_matmul(bsr:torch.Tensor, bsc:torch.Tensor):
    assert bsr.shape[-1] == bsc.shape[-2]
    assert bsr.layout == torch.sparse_bsr or bsr.layout == torch.sparse_csr
    assert bsc.layout == torch.sparse_bsc or bsc.layout == torch.sparse_csc
    crow_indices = bsr.crow_indices() # b + 1 dimensional
    col_indices = bsr.col_indices() # b + 1 dimensional
    csr_values = bsr.values() # 1 + 2 dimensional

    ccol_indices = bsc.ccol_indices() # b + 1 dimensional
    row_indices = bsc.row_indices() # b + 1 dimensional
    csc_values = bsc.values() # 1 + 2 dimensional

    idx_dtype = crow_indices.dtype

    assert bsr.ndim == 2 and bsc.ndim == 2, "bsr and bsc must be 2 dimensional. \
    batch dimension is yet not supported."
    m, n, p = bsr.shape[-2], bsr.shape[-1], bsc.shape[-1]
    dm, dn, dp = csr_values.shape[-2], csr_values.shape[-1], csc_values.shape[-1]
    sm, sn, sp = m // dm, n // dn, p // dp
    assert dm * sm == m
    assert dn * sn == n
    assert dp * sp == p

    result_step: int = 0
    coo_indices: List[int] = list()
    index: List[int] = list()
    source: List[int] = list()
    for i in range(sm):
        for j in range(sp):
            nz: bool = False
            k2 = int(ccol_indices[j].item())
            for k1 in range(int(crow_indices[i].item()), int(crow_indices[i+1].item())):
                if k2 == ccol_indices[j+1]:
                    break
                while row_indices[k2] < col_indices[k1] and k2 < ccol_indices[j+1] - 1:
                    k2 += 1
                if row_indices[k2] == col_indices[k1]:
                    index.append(result_step)
                    source.append(k1)
                    source.append(k2)
                    nz = True
            if nz:
                result_step += 1
                coo_indices.append(i)
                coo_indices.append(j)
    source = torch.tensor(source, dtype=idx_dtype, device=bsr.device).view(-1, 2)
    index = torch.tensor(index, dtype=idx_dtype, device=bsr.device)
    prod = torch.bmm(csr_values[source[:, 0]], csc_values[source[:, 1]])
    values_shape = (result_step, dm, dp)
    reduced = torch.zeros(values_shape, dtype=prod.dtype, device=prod.device)
    reduced.scatter_add_(0, index.unsqueeze(-1).unsqueeze(-1).expand_as(prod), prod)
    # Indices should be prepared on CPU, IN ANY CASE
    coo_indices = torch.tensor(coo_indices, dtype=idx_dtype, device='cpu')
    coo_indices = coo_indices.view(-1, 2).T
    # use fake coo
    dummy_val = torch.zeros(coo_indices.shape[-1], dtype=prod.dtype, device='cpu')
    dummy = torch.sparse_coo_tensor(indices=coo_indices, values=dummy_val, size=(sm, sp))
    dummy_csr = dummy.coalesce().to_sparse_csr()
    crow = dummy_csr.crow_indices().to(bsr.device)
    col = dummy_csr.col_indices().to(bsr.device)
    return torch.sparse_bsr_tensor(crow, col, reduced, size=(m, p), dtype=reduced.dtype)


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    sparse_lib = Library('aten', 'IMPL')
    sparse_lib.impl('mm', _sparse_csr_mm, 'SparseCsrCPU')
    sparse_lib.impl('mm', _sparse_csr_mm, 'SparseCsrCUDA')
    sparse_lib.impl('diagonal', diagonal_op_, 'SparseCsrCPU')
    sparse_lib.impl('diagonal', diagonal_op_, 'SparseCsrCUDA')
