
import cupy as cp
import cupyx as cpx
import torch

from torch.utils.dlpack import ( to_dlpack, from_dlpack )

from .sparse_block_matrix import ( SparseBlockMatrix, INDEX_TYPE, FLOAT_TYPE )

# from functorch import vmap

def flatten_index_from_sbm(sbm):
    '''
    Flatten the index from a SparseBlockMatrix.
    '''
    
    # Alias of the shape of the block.
    bh, bw = sbm.block_shape

    # Meshgrid coordinates for a single block.
    x = torch.arange( bw, dtype=INDEX_TYPE, device=sbm.device )
    y = torch.arange( bh, dtype=INDEX_TYPE, device=sbm.device )
    xx, yy = torch.meshgrid( x, y, indexing='xy')

    # Repeat xx and yy.
    N = sbm.n_nz_blocks
    xx = xx.repeat( N, 1, 1 )
    yy = yy.repeat( N, 1, 1 )

    # Shift xx and yy.
    xx = xx + sbm.block_indices[:, 1].view( ( N, 1, 1 ) ) * bw
    yy = yy + sbm.block_indices[:, 0].view( ( N, 1, 1 ) ) * bh

    return xx.view((-1, )), yy.view((-1, ))

def torch_to_cupy(t):
    return cp.from_dlpack( to_dlpack(t) )

def cupy_to_torch(c):
    return from_dlpack( c.toDlpack() )

def sbm_to_cupy(sbm):
    '''
    Convert a SparseBlockMatrix to a CuPy sparse matrix.
    '''

    assert sbm.is_cuda, f'Only supports Tensors on GPU. '

    # Flatten the index.
    xx, yy = flatten_index_from_sbm(sbm)

    # Flatten the data.
    data = sbm.block_storage.view((-1, ))

    # Convert xx, yy, and data to CuPy.
    xx   = torch_to_cupy(xx)
    yy   = torch_to_cupy(yy)
    data = torch_to_cupy(data)

    # Create the CuPy sparse matrix in the CSR format.
    return cpx.scipy.sparse.csr_matrix( 
        (data, (yy, xx)), 
        shape=sbm.shape,
        copy=False )