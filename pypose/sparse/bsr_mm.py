import torch
import triton
import triton.language as tl


@triton.jit
def bsr_mm_count_coeffs(
    y_ncol, z_nnz,
    x_offsets_ptr, x_columns_ptr,
    y_offsets_ptr, y_columns_ptr,
    row_min_ptr, block_counts_ptr,
    num_block_rows: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_ITERS: tl.constexpr,
):
    r'''
    For each X block-row, this kernel counts how many symbolic candidate output blocks can be produced by summing the nonzero-block counts of the corresponding Y rows referenced by X’s column indices.
    It writes the per-X-block candidate counts into block_counts[x_block + 1] and also tracks the minimum and maximum candidate column to estimate the contiguous column span for the row.
    If the row is a deep product (row_count > col_range_size), it switches to a compact range form by storing row_min[row] = min_col, storing the span size into block_counts[x_end], clearing most per-block counts to avoid redundant expansion, and it writes z_nnz once into block_counts[0] as a prefix-sum offset.

    Args:
        y_ncol:
            Number of block-columns in Y (used as a sentinel for initializing min_col).
        z_nnz:
            Number of pre-existing output blocks (written into block_counts[0] as a global prefix-sum offset).
        x_offsets_ptr:
            Pointer to X row offsets (crow_indices), length num_block_rows + 1.
        x_columns_ptr:
            Pointer to X column indices (col_indices), length x_nnz.
        y_offsets_ptr:
            Pointer to Y row offsets (crow_indices), length y_ncol + 1.
        y_columns_ptr:
            Pointer to Y column indices (col_indices), length y_nnz; assumed sorted per Y row.
        row_min_ptr:
            Output array of length num_block_rows; stores -1 for list-mode or the minimum column for range-mode.
        block_counts_ptr:
            Output array of length x_nnz + 1; stores per-X-block candidate counts at index x_block + 1 and may store a range size at x_end in range-mode; index 0 stores z_nnz.
        num_block_rows:
            Number of block-rows in X (and in the output).
        BLOCK:
            Vector width processed per loop iteration.
        NUM_ITERS:
            Number of static iterations such that NUM_ITERS * BLOCK covers the maximum nnz per X row.

    Returns:
        None. Writes results into row_min_ptr and block_counts_ptr.
    '''

    row = tl.program_id(0)

    if row >= num_block_rows:
        return

    x_beg = tl.load(x_offsets_ptr + row).to(tl.int32)
    x_end = tl.load(x_offsets_ptr + row + 1).to(tl.int32)
    row_count = tl.full((), 0, tl.int32)
    min_col = tl.full((), y_ncol, tl.int32)
    max_col = tl.full((), 0, tl.int32)
    vector_offsets = tl.arange(0, BLOCK)

    for iteration in tl.static_range(NUM_ITERS):
        x_block_ids = x_beg + iteration * BLOCK + vector_offsets
        mask = x_block_ids < x_end
        x_col = tl.load(x_columns_ptr + x_block_ids, mask=mask, other=0).to(tl.int32)
        y_beg = tl.load(y_offsets_ptr + x_col, mask=mask, other=0).to(tl.int32)
        y_end = tl.load(y_offsets_ptr + x_col + 1, mask=mask, other=0).to(tl.int32)
        block_count = (y_end - y_beg).to(tl.int32)
        tl.store(block_counts_ptr + x_block_ids + 1, block_count, mask=mask)
        row_count += tl.sum(tl.where(mask, block_count, 0), axis=0).to(tl.int32)
        has_valid_col = mask & (block_count != 0)
        first_col = tl.load(y_columns_ptr + y_beg, mask=has_valid_col, other=y_ncol).to(tl.int32)
        last_col = tl.load(y_columns_ptr + (y_end - 1), mask=has_valid_col, other=0).to(tl.int32)
        min_col = tl.minimum(min_col, tl.min(first_col, axis=0))
        max_col = tl.maximum(max_col, tl.max(last_col, axis=0))

    col_range_size = tl.maximum(0, max_col - min_col + 1)
    is_deep_product = row_count > col_range_size
    tl.store(row_min_ptr + row, tl.where(is_deep_product, min_col, -1))
    tl.store(block_counts_ptr + x_end, col_range_size, mask=is_deep_product)

    for iteration in tl.static_range(NUM_ITERS):
        x_block_ids = x_beg + iteration * BLOCK + vector_offsets
        mask = x_block_ids < (x_end - 1)
        tl.store(block_counts_ptr + x_block_ids + 1, 0, mask=is_deep_product & mask)

    tl.store(block_counts_ptr + 0, z_nnz, mask=(row == 0))


@triton.jit
def bsr_mm_list_coeffs(
    copied_z_nnz,
    mm_nnz,
    x_nrow: tl.constexpr,
    x_nnz: tl.constexpr,
    x_offsets_ptr, x_columns_ptr,
    y_offsets_ptr, y_columns_ptr,
    mm_row_min_ptr, mm_offsets_ptr,
    mm_rows_ptr, mm_cols_ptr, mm_src_blocks_ptr,
):
    r'''
    Each program instance maps a flattened candidate id mm_block into its owning X block by binary-searching mm_offsets, then computes pos_in_xblock and maps that X block index back to its X block-row by binary-searching x_offsets.
    In mm_row_min[row] == -1, it reads the output column from Y’s explicit column list at Y_offsets[k] + pos_in_xblock (where k = X_columns[x_block]) and records src_block = x_block.
    In range-mode (mm_row_min[row] != -1), it generates the output column as mm_row_min[row] + pos_in_xblock with src_block = -1, and invalid candidates are written as -1 values.

    Args:
        copied_z_nnz:
            Starting offset for candidate enumeration when some Z blocks are already accounted for.
        mm_nnz:
            Total length of the candidate arrays; bounds mm_block and masks stores.
        x_nrow:
            Number of block-rows in X.
        x_nnz:
            Total number of non-zero blocks in X (equal to x_offsets[x_nrow]).
        x_offsets_ptr:
            Pointer to X row offsets (crow_indices), length x_nrow + 1.
        x_columns_ptr:
            Pointer to X column indices (col_indices), length x_nnz.
        y_offsets_ptr:
            Pointer to Y row offsets (crow_indices), length y_ncol + 1.
        y_columns_ptr:
            Pointer to Y column indices (col_indices), length y_nnz.
        mm_row_min_ptr:
            Per-row mode selector/min-column array from the count stage; -1 means list-mode, otherwise range-mode.
        mm_offsets_ptr:
            Prefix sums over X blocks (typically cumsum(block_counts)), length x_nnz + 1.
        mm_rows_ptr:
            Output candidate rows array of length mm_nnz.
        mm_cols_ptr:
            Output candidate columns array of length mm_nnz.
        mm_src_blocks_ptr:
            Output candidate source-X-block array of length mm_nnz; set to x_block in list-mode or -1 otherwise.

    Returns:
        None. Writes results into mm_rows_ptr, mm_cols_ptr, and mm_src_blocks_ptr.
    '''
    pid = tl.program_id(0).to(tl.int32)
    mm_block = pid + copied_z_nnz.to(tl.int32)
    in_range = mm_block < mm_nnz
    candidate_id = mm_block
    x_block_search_low = tl.full((), 0, tl.int32)
    x_block_search_high = tl.full((), x_nnz, tl.int32)
    total_candidates = tl.load(mm_offsets_ptr + x_nnz).to(tl.int32)
    candidate_out_of_range = candidate_id >= total_candidates

    for _ in range(32):
        mid = (x_block_search_low + x_block_search_high) // 2
        mid_val = tl.load(mm_offsets_ptr + mid).to(tl.int32)
        go_right = mid_val <= candidate_id
        x_block_search_low = tl.where(go_right, mid, x_block_search_low)
        x_block_search_high = tl.where(go_right, x_block_search_high, mid)

    x_block_found = tl.minimum(x_block_search_low, x_nnz - 1)
    x_block_candidate_begin = tl.load(mm_offsets_ptr + x_block_found).to(tl.int32)
    x_block_candidate_end = tl.load(mm_offsets_ptr + x_block_found + 1).to(tl.int32)

    has_x_block = (
        in_range
        & (~candidate_out_of_range)
        & (x_block_candidate_begin <= candidate_id)
        & (candidate_id < x_block_candidate_end)
    )

    x_block_id = tl.where(has_x_block, x_block_found, -1)
    x_block_safe = tl.maximum(x_block_id, 0)
    x_block_begin_offset = tl.load(mm_offsets_ptr + x_block_safe, mask=has_x_block, other=0).to(tl.int32)
    pos_in_xblock = (candidate_id - x_block_begin_offset).to(tl.int32)
    row_search_low = tl.full((), 0, tl.int32)
    row_search_high = tl.full((), x_nrow, tl.int32)
    x_nnz_total = tl.load(x_offsets_ptr + x_nrow).to(tl.int32)
    xblock_oob = x_block_safe >= x_nnz_total

    for _ in range(32):
        mid = (row_search_low + row_search_high) // 2
        mid_val = tl.load(x_offsets_ptr + mid).to(tl.int32)
        go_right = mid_val <= x_block_safe
        row_search_low = tl.where(go_right, mid, row_search_low)
        row_search_high = tl.where(go_right, row_search_high, mid)

    row_found = tl.minimum(row_search_low, x_nrow - 1)
    row_xblock_begin = tl.load(x_offsets_ptr + row_found).to(tl.int32)
    row_xblock_end = tl.load(x_offsets_ptr + row_found + 1).to(tl.int32)

    has_row = (
        has_x_block
        & (~xblock_oob)
        & (row_xblock_begin <= x_block_safe)
        & (x_block_safe < row_xblock_end)
    )

    row_id = tl.where(has_row, row_found, -1)
    row_safe = tl.maximum(row_id, 0)
    row_min_col = tl.load(mm_row_min_ptr + row_safe, mask=has_row, other=-1).to(tl.int32)
    use_list_mode = has_row & (row_min_col == -1)
    x_col = tl.load(x_columns_ptr + x_block_safe, mask=use_list_mode, other=0).to(tl.int32)
    y_row_begin = tl.load(y_offsets_ptr + x_col, mask=use_list_mode, other=0).to(tl.int32)
    y_block_id = (y_row_begin + pos_in_xblock).to(tl.int32)
    col_from_list = tl.load(y_columns_ptr + y_block_id, mask=use_list_mode, other=-1).to(tl.int32)
    col_from_range = (row_min_col + pos_in_xblock).to(tl.int32)
    col_id = tl.where(use_list_mode, col_from_list, tl.where(has_row, col_from_range, -1))
    src_block_id = tl.where(use_list_mode, x_block_safe, -1)

    tl.store(mm_rows_ptr + mm_block, row_id, mask=in_range)
    tl.store(mm_cols_ptr + mm_block, col_id, mask=in_range)
    tl.store(mm_src_blocks_ptr + mm_block, src_block_id, mask=in_range)


@triton.jit
def bsr_mm_numeric_kernel(
    x_offsets_ptr, x_columns_ptr, x_values_ptr,
    y_offsets_ptr, y_columns_ptr, y_values_ptr,
    z_offsets_ptr, z_columns_ptr, z_values_ptr,
    x_block_row_num: tl.constexpr, x_block_col_num: tl.constexpr, y_block_col_num: tl.constexpr,
    x_block_row_num_pow2: tl.constexpr, x_block_col_num_pow2: tl.constexpr, y_block_col_num_pow2: tl.constexpr,
    z_block_row_num: tl.constexpr,
    DTYPE: tl.constexpr,
):
    r'''
    Performing block-sparse row (BSR) matrix multiplication on the GPU.
    This kernel computes the numeric result of z = x @ y.
    x, y, and z are stored in BSR format, and each kernel instance (tl.program_id(0)) computes one block-row of z by iterating over the precomputed sparsity pattern of that row and accumulating dense block products for matching intermediate k indices.

    Args:
        x_offsets_ptr:
            Pointer to row offsets (crow_indices) of x in BSR format.
        x_columns_ptr:
            Pointer to column indices (col_indices) of x in BSR format.
        x_values_ptr:
            Pointer to block values of x, stored as contiguous blocks (flattened) with per-block layout matching the indexing used in this kernel.
        y_offsets_ptr:
            Pointer to row offsets (crow_indices) of y in BSR format.
        y_columns_ptr:
            Pointer to column indices (col_indices) of y in BSR format.
        y_values_ptr:
            Pointer to block values of y, stored as contiguous blocks (flattened) with per-block layout matching the indexing used in this kernel.
        z_offsets_ptr:
            Pointer to row offsets (crow_indices) of z in BSR format; this is the precomputed output structure.
        z_columns_ptr:
            Pointer to column indices (col_indices) of z in BSR format; this is the precomputed output structure.
        z_values_ptr:
            Pointer to output block values of z (flattened output buffer).
        x_block_row_num:
            Number of rows in each block of x (block height).
        x_block_col_num:
            Number of columns in each block of x (and rows of y).
        y_block_col_num:
            Number of columns in each block of y (block width).
        x_block_row_num_pow2:
            Next power-of-two ≥ x_block_row_num, used for tiling.
        x_block_col_num_pow2:
            Next power-of-two ≥ x_block_col_num, used for tiling.
        y_block_col_num_pow2:
            Next power-of-two ≥ y_block_col_num, used for tiling.
        z_block_row_num:
            Number of block-rows in z (and in x).
        DTYPE:
            Triton dtype used for accumulation/computation (e.g., tl.float32).

    Returns:
        None. Writes the numeric block results in-place into z_values_ptr.
    '''
    program_id_m = tl.program_id(0)
    i = program_id_m

    if i >= z_block_row_num:
        return

    z_row_start_idx = tl.load(z_offsets_ptr + i)
    z_row_end_idx = tl.load(z_offsets_ptr + i + 1)
    x_row_start_idx = tl.load(x_offsets_ptr + i)
    x_row_end_idx = tl.load(x_offsets_ptr + i + 1)

    x_block_row_indices = tl.arange(0, x_block_row_num_pow2)
    x_block_col_indices = tl.arange(0, x_block_col_num_pow2)
    x_block_offsets = x_block_row_indices[:, None] * x_block_col_num + x_block_col_indices[None, :]
    x_mask = (x_block_row_indices[:, None] < x_block_row_num) & (x_block_col_indices[None, :] < x_block_col_num)

    y_block_row_indices = tl.arange(0, x_block_col_num_pow2)
    y_block_col_indices = tl.arange(0, y_block_col_num_pow2)
    y_block_offsets = y_block_row_indices[:, None] * y_block_col_num + y_block_col_indices[None, :]
    y_mask = (y_block_row_indices[:, None] < x_block_col_num) & (y_block_col_indices[None, :] < y_block_col_num)

    z_block_row_indices = tl.arange(0, x_block_row_num_pow2)
    z_block_col_indices = tl.arange(0, y_block_col_num_pow2)
    z_block_offsets = z_block_row_indices[:, None] * y_block_col_num + z_block_col_indices[None, :]
    z_mask = (z_block_row_indices[:, None] < x_block_row_num) & (z_block_col_indices[None, :] < y_block_col_num)

    x_block_size = x_block_row_num * x_block_col_num
    y_block_size = x_block_col_num * y_block_col_num
    z_block_size = x_block_row_num * y_block_col_num

    for z_idx in range(z_row_start_idx, z_row_end_idx):
        j = tl.load(z_columns_ptr + z_idx)
        accumulation = tl.zeros((x_block_row_num_pow2, y_block_col_num_pow2), dtype=DTYPE)

        for kx_idx in range(x_row_start_idx, x_row_end_idx):
            k = tl.load(x_columns_ptr + kx_idx)
            y_row_start_idx = tl.load(y_offsets_ptr + k)
            y_row_end_idx = tl.load(y_offsets_ptr + k + 1)

            for ky_idx in range(y_row_start_idx, y_row_end_idx):
                y_col = tl.load(y_columns_ptr + ky_idx)

                if y_col == j:
                    x_ik_ptr = x_values_ptr + (kx_idx * x_block_size) + x_block_offsets
                    x_ik = tl.load(x_ik_ptr, mask=x_mask, other=0.0)

                    y_kj_ptr = y_values_ptr + (ky_idx * y_block_size) + y_block_offsets
                    y_kj = tl.load(y_kj_ptr, mask=y_mask, other=0.0)

                    accumulation += tl.dot(x_ik, y_kj, allow_tf32=False)

        z_ij_ptr = z_values_ptr + (z_idx * z_block_size) + z_block_offsets
        tl.store(z_ij_ptr, accumulation, mask=z_mask)


def bsr_mm_triton(
    x_offsets, x_cols, x_vals,
    y_offsets, y_cols, y_vals,
    x_num_block_rows, x_num_block_cols, y_num_block_cols,
    *,
    COUNT_BLOCK: int = 128,
    LIST_NUM_WARPS: int = 4,
):
    device = x_vals.device
    x_nrow = x_num_block_rows
    x_nnz = int(x_offsets[-1].item())
    y_ncol = y_num_block_cols
    z_nnz = 0
    copied_z_nnz = 0
    row_min = torch.empty((x_nrow,), device=device, dtype=torch.int32)
    block_counts = torch.empty((x_nnz + 1,), device=device, dtype=torch.int32)
    row_nnz = (x_offsets[1:] - x_offsets[:-1]).to(torch.int64)
    max_row_nnz = int(row_nnz.max().item()) if x_nrow > 0 else 0
    NUM_ITERS = triton.cdiv(max_row_nnz, COUNT_BLOCK)
    grid_count = (x_nrow,)
    bsr_mm_count_coeffs[grid_count](
        y_ncol, z_nnz,
        x_offsets, x_cols,
        y_offsets, y_cols,
        row_min, block_counts,
        num_block_rows=x_nrow,
        BLOCK=COUNT_BLOCK,
        NUM_ITERS=NUM_ITERS,
        num_warps=4,
    )

    mm_offsets_i64 = torch.cumsum(block_counts.to(torch.int64), dim=0)
    mm_nnz_total_i64 = mm_offsets_i64[-1].item()

    if mm_nnz_total_i64 > (2**31 - 1):
        raise RuntimeError(f"mm_nnz_total too large for int32 offsets: {mm_nnz_total_i64}")

    mm_offsets = mm_offsets_i64.to(torch.int32)
    mm_nnz = int(mm_nnz_total_i64)
    mm_rows = torch.empty((mm_nnz,), device=device, dtype=torch.int32)
    mm_cols = torch.empty((mm_nnz,), device=device, dtype=torch.int32)
    mm_src = torch.empty((mm_nnz,), device=device, dtype=torch.int32)

    grid_list = (mm_nnz - copied_z_nnz,)
    bsr_mm_list_coeffs[grid_list](
        copied_z_nnz, mm_nnz,
        x_nrow=x_nrow,
        x_nnz=x_nnz,
        x_offsets_ptr=x_offsets, x_columns_ptr=x_cols,
        y_offsets_ptr=y_offsets, y_columns_ptr=y_cols,
        mm_row_min_ptr=row_min, mm_offsets_ptr=mm_offsets,
        mm_rows_ptr=mm_rows, mm_cols_ptr=mm_cols, mm_src_blocks_ptr=mm_src,
        num_warps=LIST_NUM_WARPS,
    )

    valid = (mm_rows >= 0) & (mm_cols >= 0)
    rows64 = mm_rows[valid].to(torch.int64)
    cols64 = mm_cols[valid].to(torch.int64)
    combined = (rows64 << 32) | (cols64 & 0xFFFFFFFF)
    unique_combined = torch.unique(combined, sorted=True)
    z_columns = (unique_combined & 0xFFFFFFFF).to(torch.int32)
    z_offsets = torch.zeros((x_nrow + 1,), device=device, dtype=torch.int32)
    final_rows = (unique_combined >> 32).to(torch.int32)
    row_counts_final = torch.bincount(final_rows, minlength=x_nrow).to(torch.int32)
    z_offsets[1:] = torch.cumsum(row_counts_final, dim=0)
    z_nnz_final = int(z_offsets[-1].item())
    x_br, x_bc = x_vals.shape[1], x_vals.shape[2]
    y_bc = y_vals.shape[2]
    z_vals = torch.zeros((z_nnz_final, x_br, y_bc), dtype=x_vals.dtype, device=device)
    grid_num = (x_nrow,)

    bsr_mm_numeric_kernel[grid_num](
        x_offsets, x_cols, x_vals,
        y_offsets, y_cols, y_vals,
        z_offsets, z_columns, z_vals,
        x_block_row_num=x_br, x_block_col_num=x_bc, y_block_col_num=y_bc,
        x_block_row_num_pow2=max(16, triton.next_power_of_2(x_br)),
        x_block_col_num_pow2=max(16, triton.next_power_of_2(x_bc)),
        y_block_col_num_pow2=max(16, triton.next_power_of_2(y_bc)),
        z_block_row_num=x_nrow,
        DTYPE=tl.float32,
    )

    return z_offsets, z_columns, z_vals


def bsr_output_to_dense_numpy(
    offsets_np, cols_np, vals_np,
    z_block_row_num, z_block_col_num,
    z_block_row_size, z_block_col_size,
):
    r'''
    This function converts a block-sparse matrix z in BSR format like (crow_indices, col_indices, values) back into a full dense matrix.
    Given z_offsets, z_columns, z_values, and the block shape, it reconstructs the dense (M×N) output by placing each non-zero block into its block-row i and block-col j's location.
    The inputs are assumed to represent z’s BSR structure where offsets delimit the block indices per block-row, cols stores each block’s column index, and vals stores the dense block payloads.

    Args:
        offsets_np:
            1D array of length (z_block_row_num + 1).
            offsets_np[i]..offsets_np[i+1] gives the range of block indices belonging to block-row i.
        cols_np:
            1D array of length z_nnz_blocks.
            cols_np[k] is the block-column index for the k-th nonzero block.
        vals_np:
            3D array of shape [z_nnz_blocks, z_block_row_size, z_block_col_size].
            vals_np[k] is the dense block stored at (i, cols_np[k]) for the appropriate i.
        z_block_row_num:
            Number of block-rows in z.
        z_block_col_num:
            Number of block-columns in z.
        z_block_row_size:
            Height (rows) of each dense block in z.
        z_block_col_size:
            Width (cols) of each dense block in z.

    Returns:
        A dense tensor representing z with shape
        (z_block_row_num * z_block_row_size, z_block_col_num * z_block_col_size).
    '''
    device = vals_np.device if torch.is_tensor(vals_np) else torch.device("cpu")
    offsets = torch.as_tensor(offsets_np, device=device)
    cols = torch.as_tensor(cols_np, device=device)
    vals = torch.as_tensor(vals_np, device=device)
    total_rows = z_block_row_num * z_block_row_size
    total_cols = z_block_col_num * z_block_col_size
    dense = torch.zeros((total_rows, total_cols), dtype=torch.float32)

    for i in range(z_block_row_num):
        start = int(offsets[i].item())
        end = int(offsets[i + 1].item())

        for val_idx in range(start, end):
            j = int(cols[val_idx].item())
            block = vals[val_idx]

            r0, r1 = i * z_block_row_size, (i + 1) * z_block_row_size
            c0, c1 = j * z_block_col_size, (j + 1) * z_block_col_size

            dense[r0:r1, c0:c1] = block

    return dense
