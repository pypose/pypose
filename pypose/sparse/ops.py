from typing import List
import warnings
import torch
from torch.library import Library, impl


@torch.jit.script
def _bsr_diag(input, offset: int=0):
    crow_indices = input.crow_indices() # b + 1 dimensional
    col_indices = input.col_indices() # b + 1 dimensional
    bsr_values = input.values() # 1 + 2 dimensional
    m, n = input.shape[-2], input.shape[-1]
    dense_m, dense_n = (bsr_values.shape[-2],
                                 bsr_values.shape[-1])
    sparse_m, sparse_n = m // dense_m, n // dense_n

    #simple case(block is square and offset is 0)
    if dense_m == dense_n and offset == 0:
        dummy_val = torch.zeros(bsr_values.shape[0])
        dummy = torch.sparse_csr_tensor(crow_indices=crow_indices,
                                        col_indices=col_indices,
                                        values=dummy_val)
        dummy_coo = dummy.to_sparse(layout=torch.sparse_coo).coalesce()

        indices = dummy_coo.indices()
        diag_indices = indices[0] == indices[1]
        values = bsr_values[diag_indices]
        n_diag_blocks = sparse_m if sparse_m < sparse_n else sparse_n
        results_shape = (n_diag_blocks, dense_m)
        results = torch.zeros(results_shape, dtype=values.dtype, device=values.device)
        results[indices[0, diag_indices]] = torch.diagonal(values, dim1=-2, dim2=-1)
        results = torch.flatten(results)
        return results

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
        return torch.addmm(
            torch.zeros([mat1.size(0), mat2.size(1)], dtype=mat2.dtype,
                device=mat2.device, layout=mat2.layout),
            mat1,
            mat2,
            beta=0.0,
            alpha=1.0)
    if (mat1.layout == torch.sparse_csc or mat1.layout == torch.sparse_csr) and\
        (mat2.layout == torch.sparse_csc or mat2.layout == torch.sparse_csr):
        return _sparse_csr_mm(mat1.to_sparse_csr(), mat2.to_sparse_csr())
    if mat1.layout == torch.sparse_csc and mat2.layout == torch.strided:
        return _sparse_csr_mm(mat1.to_sparse_csr(), mat2)
    if mat2.layout == torch.strided:
        return torch.addmm(
            torch.zeros([mat1.size(0), mat2.size(1)], dtype=mat1.dtype,
                device=mat1.device, layout=mat2.layout),
            mat1,
            mat2,
            beta=0.0,
            alpha=1.0)
    return torch.addmm(
        torch.zeros([mat1.size(0), mat2.size(1)], dtype=mat1.dtype, device=mat1.device,
        layout=mat1.layout),
        mat1,
        mat2,
        beta=0.0,
        alpha=1.0)

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

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    sparse_lib = Library('aten', 'IMPL')
    sparse_lib.impl('mm', _sparse_csr_mm, 'SparseCsrCPU')
    sparse_lib.impl('mm', _sparse_csr_mm, 'SparseCsrCUDA')
    sparse_lib.impl('diagonal', _bsr_diag, 'SparseCsrCPU')
    sparse_lib.impl('diagonal', _bsr_diag, 'SparseCsrCUDA')
