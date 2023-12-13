from typing import List
import torch
from torch.library import Library, impl
sparse_lib = Library('aten', 'IMPL')

# @impl(sparse_lib, 'mm', 'Sparse')
@impl(sparse_lib, 'mm', 'SparseCsrCPU')
def mm(input, other):
    if isinstance(input, torch.Tensor) and input.layout == torch.sparse_bsr:
        if isinstance(other, torch.Tensor) and other.layout == torch.sparse_bsc:
            return bsr_bsc_matmul(input, other)
    elif isinstance(input, torch.Tensor) and input.layout == torch.sparse_bsc:
        if isinstance(other, torch.Tensor) and other.layout == torch.sparse_bsr:
            return bsr_bsc_matmul(other.mT, input.mT).mT
    return input.matmul(other)

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
    dense_m, dense_n, dense_p = (csr_values.shape[-2],
                                 csr_values.shape[-1],
                                 csc_values.shape[-1])
    sparse_m, sparse_n, sparse_p = m // dense_m, n // dense_n, p // dense_p
    assert dense_m * sparse_m == m
    assert dense_n * sparse_n == n
    assert dense_p * sparse_p == p

    result_step: int = 0
    coo_indices: List[int] = list()
    index: List[int] = list()
    source: List[int] = list()
    for i in range(sparse_m):
        for j in range(sparse_p):
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
    values_shape = (result_step, dense_m, dense_p)
    reduced = torch.zeros(values_shape, dtype=prod.dtype, device=prod.device)
    reduced.scatter_add_(0, index.unsqueeze(-1).unsqueeze(-1).expand_as(prod), prod)
    coo_indices = torch.tensor(coo_indices, dtype=idx_dtype, device=bsr.device)
    coo_indices = coo_indices.view(-1, 2).T
    # return torch.sparse_coo_tensor(indices=result_indices,
    #                                values=reduced,
    #                                size=(sparse_m, sparse_p, dense_m, dense_p)).coalesce()
    # use fake coo
    dummy_val = torch.zeros(coo_indices.shape[-1], dtype=prod.dtype, device=prod.device)
    dummy = torch.sparse_coo_tensor(indices=coo_indices,
                                    values=dummy_val,
                                    size=(sparse_m, sparse_p)).coalesce()
    dummy_csr = dummy.to_sparse_csr()
    return torch.sparse_bsr_tensor(dummy_csr.crow_indices(),
                                   dummy_csr.col_indices(),
                                   reduced,
                                   size=(m, p), dtype=reduced.dtype)
