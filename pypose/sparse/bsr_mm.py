import torch
import numpy as np
import triton
import triton.language as tl
import random

device = "cuda" if torch.cuda.is_available() else "cpu"

@triton.jit
def bsr_mm_numeric_kernel(
    A_offsets_ptr, A_columns_ptr, A_values_ptr,
    B_offsets_ptr, B_columns_ptr, B_values_ptr,
    C_offsets_ptr, C_columns_ptr, C_values_ptr,
    A_block_row_num: tl.constexpr, A_block_col_num: tl.constexpr, B_block_col_num: tl.constexpr,
    A_block_row_num_pow2: tl.constexpr, A_block_col_num_pow2: tl.constexpr, B_block_col_num_pow2: tl.constexpr,
    C_block_row_num: tl.constexpr,
    DTYPE: tl.constexpr,
):
    r'''
    Performs block–sparse row (BSR) matrix multiplication on the GPU.

    This kernel computes the numeric result of

        C = A @ B

    where A, B, and C are stored in BSR format. Each kernel instance
    (identified by tl.program_id(0)) is responsible for computing one
    block row of C.

    Args:
        A_offsets_ptr: pointer to row offsets (crow_indices) of A in BSR format
        A_columns_ptr: pointer to column indices (col_indices) of A in BSR format
        A_values_ptr: pointer to block values of A, flattened to 1D
        B_offsets_ptr: pointer to row offsets (crow_indices) of B in BSR format
        B_columns_ptr: pointer to column indices (col_indices) of B in BSR format
        B_values_ptr: pointer to block values of B, flattened to 1D
        C_offsets_ptr: pointer to row offsets of C (pre-computed structure)
        C_columns_ptr: pointer to column indices of C (pre-computed structure)
        C_values_ptr: pointer to block values of C, flattened to 1D (output buffer)
        A_block_row_num: number of rows in each block of A (block height)
        A_block_col_num: number of columns in each block of A (and rows of B)
        B_block_col_num: number of columns in each block of B (block width)
        A_block_row_num_pow2: next power-of-two ≥ A_block_row_num, used for tiling
        A_block_col_num_pow2: next power-of-two ≥ A_block_col_num
        B_block_col_num_pow2: next power-of-two ≥ B_block_col_num
        C_block_row_num: number of block rows in C (and A)
        DTYPE: Triton dtype used for computation (e.g. tl.float32)

    Returns:
        None. The kernel does not return a value; it writes all results in-place
        into C_values_ptr.
    '''

    program_id_m = tl.program_id(0)
    i = program_id_m

    if i >= C_block_row_num:
        return

    C_row_start_idx = tl.load(C_offsets_ptr + i)
    C_row_end_idx = tl.load(C_offsets_ptr + i + 1)
    A_row_start_idx = tl.load(A_offsets_ptr + i)
    A_row_end_idx = tl.load(A_offsets_ptr + i + 1)

    A_block_row_indices = tl.arange(0, A_block_row_num_pow2)
    A_block_col_indices = tl.arange(0, A_block_col_num_pow2)
    A_block_offsets = A_block_row_indices[:, None] * A_block_col_num + A_block_col_indices[None, :]
    A_mask = (A_block_row_indices[:, None] < A_block_row_num) & (A_block_col_indices[None, :] < A_block_col_num)

    B_block_row_indices = tl.arange(0, A_block_col_num_pow2)
    B_block_col_indices = tl.arange(0, B_block_col_num_pow2)
    B_block_offsets = B_block_row_indices[:, None] * B_block_col_num + B_block_col_indices[None, :]
    B_mask = (B_block_row_indices[:, None] < A_block_col_num) & (B_block_col_indices[None, :] < B_block_col_num)

    C_block_row_indices = tl.arange(0, A_block_row_num_pow2)
    C_block_col_indices = tl.arange(0, B_block_col_num_pow2)
    C_block_offsets = C_block_row_indices[:, None] * B_block_col_num + C_block_col_indices[None, :]
    C_mask = (C_block_row_indices[:, None] < A_block_row_num) & (C_block_col_indices[None, :] < B_block_col_num)

    A_block_size = A_block_row_num * A_block_col_num
    B_block_size = A_block_col_num * B_block_col_num
    C_block_size = A_block_row_num * B_block_col_num

    for z_idx in range(C_row_start_idx, C_row_end_idx):
        j = tl.load(C_columns_ptr + z_idx)
        accumulation = tl.zeros((A_block_row_num_pow2, B_block_col_num_pow2), dtype=DTYPE)

        for ka_idx in range(A_row_start_idx, A_row_end_idx):
            k = tl.load(A_columns_ptr + ka_idx)
            B_row_start_idx = tl.load(B_offsets_ptr + k)
            B_row_end_idx = tl.load(B_offsets_ptr + k + 1)

            for kb_idx in range(B_row_start_idx, B_row_end_idx):
                B_col = tl.load(B_columns_ptr + kb_idx)

                if B_col == j:
                    A_ik_ptr = A_values_ptr + (ka_idx * A_block_size) + A_block_offsets
                    A_ik = tl.load(A_ik_ptr, mask=A_mask, other=0.0)

                    B_kj_ptr = B_values_ptr + (kb_idx * B_block_size) + B_block_offsets
                    B_kj = tl.load(B_kj_ptr, mask=B_mask, other=0.0)

                    accumulation += tl.dot(A_ik, B_kj, allow_tf32=False)

        C_ij_ptr = C_values_ptr + (z_idx * C_block_size) + C_block_offsets
        tl.store(C_ij_ptr, accumulation, mask=C_mask)

def bsr_mm_outcome_structure(
    A_offsets, A_cols,
    B_offsets, B_cols,
    A_block_row_num,
):
    r'''
    Builds the sparsity pattern (structure) of the C = A @ B result in BSR format.

    Given the BSR row offsets and column indices of A and B (A_offsets/A_cols, B_offsets/B_cols), this function figures out:

        - which block-columns j are non-zero in each block-row i of C, and
        - how many non-zero blocks C will have in total.

    It does NOT compute any numeric values. it only builds the structure for the outcome C.

    Args:
        A_offsets: 1D int tensor (crow_indices of A), shape [num_block_rows_A + 1]. A_offsets[i]..A_offsets[i+1] is the range of block indices
        A_cols: 1D int tensor (col_indices of A), shape [nnz_blocks_A]. Column index for each non-zero block of A.
        B_offsets: 1D int tensor (crow_indices of B), shape [num_block_rows_B + 1]
        B_cols: 1D int tensor (col_indices of B), shape [nnz_blocks_B]. Column index for each non-zero block of B.
        A_block_row_num: number of block-rows in A (and C). We iterate i=0..A_block_row_num-1.

    Returns:
        C_offsets: 1D int tensor, shape [A_block_row_num + 1].
        C_columns: 1D int tensor, shape [C_all_block_nums]. Column indices (col_indices) for all non-zero blocks of C.
        C_all_block_nums: int, total number of non-zero blocks in C used to size the C values array

    This function is used before launching the Triton kernel: it allocates the
    correct-size storage for C’s block values and tells the kernel exactly where
    each output block (i, j) should be written.
    '''

    device = A_offsets.device
    A_offsets_cpu = A_offsets.cpu().numpy()
    A_cols_cpu = A_cols.cpu().numpy()
    B_offsets_cpu = B_offsets.cpu().numpy()
    B_cols_cpu = B_cols.cpu().numpy()
    C_cols_list = []
    C_offsets_list = [0]

    for i in range(A_block_row_num):
        cols_j_in_row_i = set()
        A_start, A_end = A_offsets_cpu[i], A_offsets_cpu[i + 1]

        for ka_idx in range(A_start, A_end):
            k = A_cols_cpu[ka_idx]
            B_start, b_end = B_offsets_cpu[k], B_offsets_cpu[k + 1]

            for kb_idx in range(B_start, b_end):
                j = B_cols_cpu[kb_idx]
                cols_j_in_row_i.add(j)

        sorted_cols_j = sorted(list(cols_j_in_row_i))
        C_cols_list.extend(sorted_cols_j)
        C_offsets_list.append(len(C_cols_list))

    C_offsets = torch.tensor(C_offsets_list, dtype=torch.int32, device=device)
    C_columns = torch.tensor(C_cols_list, dtype=torch.int32, device=device)
    C_all_block_nums = len(C_cols_list)

    return C_offsets, C_columns, C_all_block_nums

def bsr_mm_triton(
    A_offsets, A_cols, A_vals,
    B_offsets, B_cols, B_vals,
    A_num_block_rows, A_num_block_cols, B_num_block_cols,
):

    r'''
    Performs block-sparse matrix multiplication C = A @ B using the Triton kernel.

    This is the high-level driver that:
    1) Reads BSR metadata of A and B (offsets, column indices, block values),
    2) Computes the sparsity pattern of C using bsr_mm_outcome_structure,
    3) Allocates storage for C’s block values,
    4) Flattens the block tensors into 1D buffers,
    5) Launches the Triton kernel (bsr_mm_numeric_kernel) to fill C_vals in-place,
    6) Returns the BSR (crow_indices, col_indices, values) of C.

    Args:
    A_offsets: 1D int tensor (crow_indices of A), shape [A_num_block_rows + 1]
    A_cols: 1D int tensor (col_indices of A), shape [nnz_blocks_A]
    A_vals: 3D tensor, shape [nnz_blocks_A, A_block_row_num, A_block_col_num]. Block values of A (row-major inside each block).

    B_offsets: 1D int tensor (crow_indices of B), shape [B_num_block_rows + 1]
    B_cols: 1D int tensor (col_indices of B), shape [nnz_blocks_B]
    B_vals: 3D tensor, shape [nnz_blocks_B, A_block_col_num, B_block_col_num]. Block values of B.

    A_num_block_rows: number of block-rows of A and C
    A_num_block_cols: number of block-columns of A and block-rows of B
    B_num_block_cols: number of block-columns of B and C (kept for completeness. C’s pattern is derived from A/B offsets)

    Returns:
    C_offsets: 1D int tensor, crow_indices of C, shape [A_num_block_rows + 1]
    C_columns: 1D int tensor, col_indices of C, shape [C_all_block_nums]
    C_vals: 3D tensor, shape [C_all_block_nums, A_block_row_num, B_block_col_num]. Block values of the result C in BSR format.
    '''


    device = A_offsets.device
    dtype = A_vals.dtype
    A_block_row_num, A_block_col_num = A_vals.shape[1], A_vals.shape[2]
    B_block_col_num = B_vals.shape[2]

    C_offsets, C_columns, C_all_block_nums = bsr_mm_outcome_structure(A_offsets, A_cols, B_offsets, B_cols, A_num_block_rows)
    C_vals = torch.empty(C_all_block_nums, A_block_row_num, B_block_col_num, dtype=dtype, device=device)

    if C_all_block_nums == 0:
        return C_offsets, C_columns, C_vals

    A_vals_flat = A_vals.reshape(-1)
    B_vals_flat = B_vals.reshape(-1)
    C_vals_flat = C_vals.reshape(-1)
    grid_dimension = (A_num_block_rows,)

    A_block_row_num_pow2 = max(16, triton.next_power_of_2(A_block_row_num))
    A_block_col_num_pow2 = max(16, triton.next_power_of_2(A_block_col_num))
    B_block_col_num_pow2 = max(16, triton.next_power_of_2(B_block_col_num))

    bsr_mm_numeric_kernel[grid_dimension](
        A_offsets, A_cols, A_vals_flat,
        B_offsets, B_cols, B_vals_flat,
        C_offsets, C_columns, C_vals_flat,
        A_block_row_num=A_block_row_num,
        A_block_col_num=A_block_col_num,
        B_block_col_num=B_block_col_num,
        A_block_row_num_pow2=A_block_row_num_pow2,
        A_block_col_num_pow2=A_block_col_num_pow2,
        B_block_col_num_pow2=B_block_col_num_pow2,
        C_block_row_num=A_num_block_rows,
        DTYPE=tl.float32,
    )

    return C_offsets, C_columns, C_vals

def bsr_output_to_dense_numpy(offsets_np, cols_np, vals_np, C_block_row_num, C_block_col_num, C_block_row_size, C_block_col_size):
    r'''
    Converts a block-sparse matrix in BSR format (crow_indices, col_indices, values)
    back into a full dense NumPy matrix.

    This function is the inverse of BSR compression:
    When row_offsets, column_indices, block_values and the block shape are given, it reconstructs
    the full M×N dense matrix by placing each non-zero block in its corresponding location.

    Args:
    offsets_np:
        1D NumPy array of length (C_block_row_num + 1).
        This is the BSR "crow_indices".
        offsets_np[i] .. offsets_np[i+1] gives the range of block indices in row-block i.

    cols_np:
        1D NumPy array of length C_all_block_nums.
        For each block index k, cols_np[k] gives the column-block index j
        where that block should be placed.

    vals_np:
        3D NumPy array of shape [C_all_block_nums, C_block_row_size, C_block_col_size].
        Each entry vals_np[k] is a dense block that fits into the dense matrix.

    C_block_row_num: Number of block-rows of the matrix C.
    C_block_col_num: Number of block-columns of the matrix C.
    C_block_row_size: Height (in rows) of each block.
    C_block_col_size: Width (in columns) of each block.

    Returns:
    A fully reconstructed dense NumPy array representing the BSR matrix.
    '''

    total_rows = C_block_row_num * C_block_row_size
    total_cols = C_block_col_num * C_block_col_size
    dense = np.zeros((total_rows, total_cols), dtype=np.float32)

    for i in range(C_block_row_num):
        start = offsets_np[i]
        end = offsets_np[i + 1]

        for val_idx in range(start, end):
            j = cols_np[val_idx]
            val_block = vals_np[val_idx]

            r0, r1 = i * C_block_row_size, (i + 1) * C_block_row_size
            c0, c1 = j * C_block_col_size, (j + 1) * C_block_col_size

            dense[r0:r1, c0:c1] = val_block

    return dense
