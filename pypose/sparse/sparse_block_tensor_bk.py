
from typing import List, Set, Dict, Tuple, Optional, Iterable, Union

import re
from numpy import block
import numpy as np
from scipy.sparse import bsr_matrix

import cupy as cp
import cupyx as cpx

import torch
from torch.utils.dlpack import ( to_dlpack, from_dlpack )

# Globals.
INDEX_TYPE = torch.int64
FLOAT_TYPE = torch.float32


_HANDLED_FUNCS_SPARSE = [
    'matmul'
]
class MyTensor(torch.Tensor):

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs={}):
        global _HANDLED_FUNCS_SPARSE

        mytypes = (torch.Tensor if t is MyTensor else t for t in types)
        myargs = (t.sbt if isinstance(t, MyTensor) else t for t in args)
        res = torch.Tensor.__torch_function__(func, mytypes, myargs, kwargs)

        print(f'func.__name__ = {func.__name__}')
        if func.__name__ in _HANDLED_FUNCS_SPARSE:
            out = MyTensor()
            out.sbt = res
        else:
            out = res

        return out


def sparse_block_tensor(indices, values, size=None, dtype=None, device=None, requires_grad=False):
    data = torch.sparse_coo_tensor(indices, values, size=size, dtype=dtype, device=device, requires_grad=requires_grad)
    x = MyTensor()
    x.sbt = data
    return x



class SparseBlockTensorNew(torch.Tensor):
    def __init__(self, *data):
        pass

    @staticmethod
    def to_sparse(data):
        '''
            this function turns tensor into sparse format
        '''
        if( not isinstance(data, torch.Tensor) ):
            raise NotImplementedError("Input of SparseBlockTensor has to be a Tensor (dense or sparse)")

        if( data.is_sparse ):
            return data
        else:
            data = data.to_sparse_coo()
            return data

    @staticmethod
    def __new__(cls, *data):
        #sbt = cls.to_sparse( data[0] ) if isinstance(data[0], torch.Tensor) else cls.to_sparse(torch.Tensor(*data))
        sbt = data[0] if isinstance(data[0], torch.Tensor) else torch.Tensor(*data)
        return torch.Tensor.as_subclass( sbt, SparseBlockTensorNew)


# ========== Mutual conversion betwen torch sparse coo tensor and SparseBlockTensor. ==========

def sbt_to_torch_sparse_coo(sbt):
    s_shape = ( *sbt.shape_blocks, *sbt.block_shape )

    indices = sbt.block_indices[:, :2].permute((1, 0))

    return torch.sparse_coo_tensor( 
        indices, sbt.block_storage, s_shape, dtype=sbt.dtype, device=sbt.device)

def torch_sparse_coo_to_sbt(s):
    assert ( s.is_sparse ), f's must be a sparse tensor. '
    assert ( s.layout == torch.sparse_coo ), f's must have the COO layout. s.layout = {s.layout}'
    assert ( s.ndim == 4 ), f's.shape == {s.shape}'

    if not s.is_coalesced():
        s = s.coalesce()
    
    shape_blocks = s.shape[:2]
    block_shape  = s.values().shape[1:3]

    sbt = SparseBlockTensor( block_shape, dtype=s.dtype, device=s.device )
    sbt.create(
        shape_blocks=shape_blocks, 
        block_indices=s.indices().detach().clone(), 
        device=s.device)
    sbt.block_storage = s.values().detach().clone()
    sbt.dtype = s.dtype
    sbt.coalesced = True

    return sbt

# ========== Mutual conversion betwen SciPy Block Sparse Row Matrix and SparseBlockTensor. ==========

def sbt_to_bsr_cpu(sbt):
    '''
    Convert a SparseBlockTensor to a Block Row Matrix defined by SciPy. 
    Note that this function is for test purpose. If sbt is on GPU, then
    all the data will be off loaded to CPU.

    NOTE: This function assumes that there are no empty rows of blocks.
    '''

    if not sbt.is_coalesced():
        sbt = sbt.coalesce()

    # We only need these two.
    block_indices = sbt.block_indices
    block_storage = sbt.block_storage

    # Compose the indptr and indices variables. 
    middle = torch.nonzero( torch.diff( block_indices[:, 0] ) ).view((-1,)) + 1
    indptr = torch.zeros( (middle.numel() + 2,), dtype=INDEX_TYPE, device=block_indices.device )
    indptr[1:-1] = middle
    indptr[-1] = block_indices.shape[0] # block_indices is a table with many rows.
    indices = block_indices[:, 1]

    assert indptr.numel() - 1 == sbt.shape_blocks[0], \
        f'indptr.numel() = {indptr.numel()}, sbt.rows = {sbt.rows}'

    if sbt.is_cuda:
        # sbt = sbt.to(device='cpu')
        block_indices = block_indices.cpu()
        block_storage = block_storage.cpu()
        indptr = indptr.cpu()
        indices = indices.cpu()
    
    return bsr_matrix( 
        ( block_storage.numpy(), indices.numpy(), indptr.numpy() ),
        shape=sbt.shape )

# DTYPE_NUMPY_TO_TORCH = {
#     np.int: torch.int,
#     np.int64: torch.int64,
#     np.float: torch.float64,
#     np.float32: torch.float32,
#     np.float64: torch.float64
# }

def get_equivalent_torch_dtype(dtype):
    '''
    Return the equivalent torch dtype based on numpy dtype.
    '''

    # Cannot use DTYPE_NUMPY_TO_TORCH for things like np.dtype('int64')
    if dtype == int:
        return torch.int # This might be wrong.
    elif dtype == np.dtype('int32'):
        return torch.int32
    elif dtype == np.dtype('int64'):
        return torch.int64
    elif dtype == float:
        return torch.float64
    elif dtype == np.dtype('float32'):
        return torch.float32
    else:
        raise Exception(f'dtype {dtype} not supported. ')

def bsr_cpu_to_sbt(bsr, device=None):
    '''
    Warning: bsr.sum_duplicates() is called such that bsr is changed.
    bsr.data may referring to an outer object. That object is also changed.

    Convert a Block Row Matrix defined by SciPy to SparseBlockTensor.

    dtype association:
    np.int     -> torch.int
    np.int64   -> torch.int64
    np.float   -> torch.float64
    np.float64 -> torch.float64
    np.float32 -> torch.float32
    '''

    # Call the sum_duplicates() of bsr.
    bsr.sum_duplicates()

    # Get the dtype.
    dtype = get_equivalent_torch_dtype(bsr.dtype)

    # Create the SparseBlockTensor.
    sbt = SparseBlockTensor( bsr.blocksize, dtype=dtype, device=device )

    # Shape of blocks.
    shape_blocks = [
        bsr.shape[0] // bsr.blocksize[0], 
        bsr.shape[1] // bsr.blocksize[1] ]

    # Block indices.
    d = np.diff( bsr.indptr )
    block_row_indices = np.repeat( np.arange(d.size, dtype=int), d )
    block_indices = np.stack( ( block_row_indices, bsr.indices ), axis=0 )
    block_indices = torch.from_numpy( block_indices )

    # Create memory for sbt.
    sbt.create(
        shape_blocks=shape_blocks,
        block_indices=block_indices,
        device=device )

    # Block storage.
    sbt.block_storage = torch.from_numpy( bsr.data ).to(device=device)

    # Coalesce.
    # The Block Row Matrix created by SciPy is always coalesced since we call
    # sum_duplicates() explicitly. 
    sbt.coalesced = True

    return sbt

# ========== Conversion from SparbBlockMatrix to CuPy array. ==========

def flatten_index_from_sbt(sbt):
    '''
    Flatten the index from a SparseBlockTensor.
    '''
    
    # Alias of the shape of the block.
    bh, bw = sbt.block_shape

    # Meshgrid coordinates for a single block.
    x = torch.arange( bw, dtype=INDEX_TYPE, device=sbt.device )
    y = torch.arange( bh, dtype=INDEX_TYPE, device=sbt.device )
    xx, yy = torch.meshgrid( x, y, indexing='xy')

    # Repeat xx and yy.
    N = sbt.n_nz_blocks
    xx = xx.repeat( N, 1, 1 )
    yy = yy.repeat( N, 1, 1 )

    # Shift xx and yy.
    xx = xx + sbt.block_indices[:, 1].view( ( N, 1, 1 ) ) * bw
    yy = yy + sbt.block_indices[:, 0].view( ( N, 1, 1 ) ) * bh

    return xx.view((-1, )), yy.view((-1, ))

def torch_to_cupy(t):
    return cp.from_dlpack( to_dlpack(t) )

def cupy_to_torch(c):
    return from_dlpack( c.toDlpack() )

def sbt_to_cupy(sbt):
    '''
    Convert a SparseBlockTensor to a CuPy sparse matrix.
    '''

    assert sbt.is_cuda, f'Only supports Tensors on GPU. '

    # Flatten the index.
    xx, yy = flatten_index_from_sbt(sbt)

    # Flatten the data.
    data = sbt.block_storage.view((-1, ))

    # Convert xx, yy, and data to CuPy.
    xx   = torch_to_cupy(xx)
    yy   = torch_to_cupy(yy)
    data = torch_to_cupy(data)

    # Create the CuPy sparse matrix in the CSR format.
    return cpx.scipy.sparse.csr_matrix( 
        (data, (yy, xx)), 
        shape=sbt.shape,
        copy=False )

# =========== The SparseBlockTensor class. ===========

class SparseBlockTensor(object):
    def __init__(self, 
        block_shape: Iterable[int],
        dtype: Optional[torch.dtype] = None,
        device: Optional[str] = None) -> None:
        super().__init__()

        '''
        Sparse block matrix using same block size for storage.
        '''

        for v in block_shape:
            assert ( v > 0 ), f'block_shape must have all positive values. block_shape = {block_shape}'
        self.block_shape = block_shape # The size of the individual blocks.

        # row_block_structure: 1D array of int containing the row layout of the blocks. 
        #     A component i of the array should contain the index of the 
        #     first row of the block i+1.
        # col_block_structure: 1D array of int containing the column layout of the blocks. 
        #     A component i of the array should contain the index of the 
        #     first column of the block i+1.

        self.row_block_structure = torch.zeros((1,), dtype=INDEX_TYPE, device=device)
        self.col_block_structure = torch.zeros((1,), dtype=INDEX_TYPE, device=device)

        self.block_indices = None # Should be 2D Tensor of ints (3 columns). The values are the indices of block column, not the value column.
                                  # Similar to g2o's _blockCols. 
        self.block_storage = None # Should be 3D Tensor.
        self.dtype = dtype if dtype is not None else FLOAT_TYPE

        self.coalesced = False

    def is_coalesced(self):
        '''
        Naive implementation.
        '''
        return self.coalesced

    def dimension_str(self):
        return \
            f'block_shape = {self.block_shape}\n' + \
            f'row_block_structure.numel() = {self.row_block_structure.numel()}\n' + \
            f'col_block_structure.numel() = {self.col_block_structure.numel()}\n' + \
            f'block_indices.shape = {self.block_indices.shape}\n' + \
            f'block_storage.shape = {self.block_storage.shape}'

    def show_dimensions(self):
        print(self.dimension_str())

    def create(self, 
        shape_blocks: Iterable[int],
        block_indices: Union[ List[List[int]], torch.Tensor ],
        device: Optional[str]=None) -> None:
        '''
        Argument definitions are copied from g2o:
        shape_blocks: number of blocks along vertical (number of block rows) and horizontal (number of block columns) directions.
        block_indices: 2D list or Tensor. 2 rows. Every colume records the block index of a single block.
        device: Tensor device.
        '''

        if device is None:
            device = self.row_block_structure.device

        self.row_block_structure = ( torch.arange( shape_blocks[0], dtype=INDEX_TYPE, device=device ) + 1 ) * self.block_shape[0]
        self.col_block_structure = ( torch.arange( shape_blocks[1], dtype=INDEX_TYPE, device=device ) + 1 ) * self.block_shape[1]

        if isinstance(block_indices, list):
            assert ( len(block_indices) == 2 ), f'block_indices must be a 2D list. len(block_indices) = {len(block_indices)}'
            block_indices = torch.Tensor(block_indices).to(device=device, dtype=INDEX_TYPE)
        elif isinstance(block_indices, torch.Tensor):
            assert ( block_indices.ndim == 2 ), f'Wrong dimension of block_indices. block_indices.shape = {block_indices.shape}'
            block_indices = block_indices.to(device=device, dtype=INDEX_TYPE)
        else:
            raise Exception(f'block_indices must be a 2D list or Tensor')

        self.block_indices = torch.zeros( (block_indices.shape[1], 3), device=device, dtype=INDEX_TYPE )
        self.block_indices[:, :2] = block_indices.permute((1, 0))
        self.block_indices[:, 2]  = torch.arange( self.block_indices.shape[0], dtype=INDEX_TYPE, device=device )
        
        self.block_storage = torch.zeros(
            (self.block_indices.shape[0], *self.block_shape), 
            dtype=self.dtype, device=device )

    def set_block_storage(self, block_storage, clone=False):
        assert ( block_storage.ndim == 3 ), f'block_storage.ndim = {block_storage.ndim}'
        assert ( block_storage.shape[0] == self.block_indices.shape[0] ), \
            f'block_storage.shape = {block_storage.shape}, self.block_indices.shape = {self.block_indices.shape}'
        assert ( block_storage.shape[1:3] == self.block_shape ), \
            f'block_storage.shape = {block_storage.shape}, self.block_shape = {self.block_shape}'

        if clone:
            self.block_storage = block_storage.clone()
        else:
            self.block_storage = block_storage

        self.dtype = self.block_storage.dtype

    @property
    def rows(self):
        n = self.row_block_structure.numel()
        # TODO: can we have better efficiency?
        return self.row_block_structure[-1].item() if n > 0 and self.row_block_structure[0] >= 1 else 0
    
    @property
    def cols(self):
        n = self.col_block_structure.numel()
        # TODO: can we have better efficiency?
        return self.col_block_structure[-1].item() if n > 0 and self.col_block_structure[0] >= 1 else 0

    @property
    def shape(self):
        return ( self.rows, self.cols )

    @property
    def shape_blocks(self):
        if self.block_indices is None:
            return (0, 0)
        
        return ( self.rows // self.block_shape[0], self.cols // self.block_shape[1] )

    @property
    def device(self):
        return self.row_block_structure.device

    @property
    def is_cuda(self):
        return self.row_block_structure.is_cuda

    @device.setter
    def device(self, d):
        self.row_block_structure = self.row_block_structure.to(device=d)
        self.col_block_structure = self.col_block_structure.to(device=d)
        
        if self.block_indices is not None:
            self.block_indices = self.block_indices.to(device=d)

        if self.block_storage is not None:
            self.block_storage = self.block_storage.to(device=d)

    @property
    def n_nz_blocks(self):
        return self.block_indices.shape[0]

    def type(self, dtype=None):
        if dtype is None:
            return self.dtype
        else:
            # There might be bugs considering that when dtype == self.dtype, self is
            # returned instead of a clone. However, this seems the same behavior with
            # PyTorch's Tensor.type() function.
            if dtype != self.dtype:
                new_matrix = self.clone()
                if new_matrix.block_storage is not None:
                    new_matrix.block_storage = self.block_storage.type(dtype=dtype)
                new_matrix.dtype = dtype
                return new_matrix
            else:
                return self

    def to(self, device=None, dtype=None, copy=False):
        assert ( dtype is not None or device is not None ), \
            f'dtype and device cannot both be None'

        if isinstance(device, str):
            if device == 'cuda':
                device = torch.device( '%s:%d' % ( 'cuda', torch.cuda.current_device() ) )
            elif re.match(r'^(cuda:\d+)', device):
                device = torch.device( device )
            elif device == 'cpu':
                device = torch.device('cpu')
            else:
                raise Exception(f'device must be cuda, cuda:x or cpu if supplied as str. device = {device}')

        # Record the event of deep copy.
        flag_copied = False

        if dtype is not None and dtype != self.dtype:
            m = self.type(dtype=dtype)
            flag_copied = True
        else:
            m = self

        if device is not None and device != self.device:
            if flag_copied:
                m.device = device # m is modified.
            else:
                m = m.clone()
                m.device = device # m is modified.
                flag_copied = True

        if copy and not flag_copied:
            return self.clone()
        else:
            return m

    def coalesce(self):
        '''
        Use PyTorch's sparse matrix to perform the coalesce operation.
        This might not be memory-efficient since a copy of the matrix is created during the operation.
        '''

        # TODO: make a flag system shownig the status of coalesce. 
        # May need also to wrap some member variables by accessor functions.

        # Convert the sparse block matrix to torch sparse matrix.
        scoo = sbt_to_torch_sparse_coo(self)

        # coalesce.
        scoo = scoo.coalesce()

        # Convert the torch sparse matrix back to the sparse block matrix.
        return torch_sparse_coo_to_sbt(scoo)

    def rows_of_block(self, idx: int):
        assert ( idx >= 0 ), \
            f'idx must be a positive integer. idx = {idx}'
        # TODO: Better efficiency?
        return self.row_block_structure[0] \
            if idx == 0 \
            else ( self.row_block_structure[idx] - self.row_block_structure[idx - 1] ).item()
    
    def cols_of_block(self, idx: int):
        assert ( idx >= 0 ), \
            f'idx must be a positive integer. idx = {idx}'
        # TODO: Better efficiency?
        return self.col_block_structure[0] \
            if idx == 0 \
            else ( self.col_block_structure[idx] - self.col_block_structure[idx - 1] ).item()

    def row_base_of_block(self, idx: int):
        assert ( idx >= 0 ), \
            f'idx must be a positive integer. idx = {idx}'
        # TODO: Better efficiency?
        return 0 if idx == 0 else self.row_block_structure[ idx - 1 ].item()

    def col_base_of_block(self, idx: int):
        assert ( idx >= 0 ), \
            f'idx must be a positive integer. idx = {idx}'
        # TODO: Better efficiency?
        return 0 if idx == 0 else self.col_block_structure[ idx - 1 ].item()

    def __deepcopy__(self):
        new_matrix = SparseBlockTensor( self.block_shape )
        new_matrix.col_block_structure = self.col_block_structure.clone()
        new_matrix.row_block_structure = self.row_block_structure.clone()
        
        if self.block_indices is not None:
            new_matrix.block_indices = self.block_indices.clone()

        if self.block_storage is not None:
            new_matrix.block_storage = self.block_storage.clone()

        new_matrix.coalesced = self.coalesced

        return new_matrix

    def clone(self):
        return self.__deepcopy__()

    def slice(self, block_rows: Iterable[int], block_cols: Iterable[int], clone: bool=False):
        '''
        Return the selected blocks.
        block_rows: the first and one-past-last row indices of blocks.
        block_cols: the first and one-past-last col indices of blocks.
        clone: True if the blocks are cloned.

        Returns:
        A new SparseBlockTensor object.
        '''
        raise NotImplementedError()

    def transpose_(self):
        '''
        In-place transpose.
        '''
        self.block_shape = self.block_shape[::-1]

        # We just need to swap row and column structures and the order of self.block_indices.
        self.row_block_structure, self.col_block_structure = self.col_block_structure, self.row_block_structure
        self.block_indices = torch.index_select( 
            self.block_indices, 
            1, 
            torch.LongTensor([1, 0, 2]).to(device=self.block_indices.device) )

        # Transpose the blocks.
        self.block_storage = self.block_storage.permute((0, 2, 1))

        self.coalesced = False

        return self

    def transpose(self):
        '''
        Return a copy of the current sparse block matrix and transpose.
        '''
        # Make a copy of the current sparse block matrix.
        new_matrix = self.clone()
        return new_matrix.transpose_()

    def add_sparse_block_matrix(self, other):
        '''
        Currently only supports adding two sparse block matrices with the same block_shape.
        '''

        assert ( self.rows == other.rows and self.cols == other.cols), \
            f'Incompatible dimensions: self: [{self.rows}, {self.cols}], other: [{other.rows}, {other.cols}]. '

        # Concatenate the raw data of the two SparseBlockTensor.
        c_block_indices = torch.cat( ( self.block_indices, other.block_indices ), dim=0 )
        c_block_storage = torch.cat( ( self.block_storage, other.block_storage ), dim=0 )

        # Perform add by converting to a PyTorch sparse tensor.
        s_shape = ( *self.shape_blocks, *self.block_shape )
        indices = c_block_indices[:, :2].permute((1, 0))

        s = torch.sparse_coo_tensor( 
            indices, c_block_storage, s_shape, 
            dtype=self.dtype, device=self.device)

        return torch_sparse_coo_to_sbt(s)

    def add_broadcast(self, other):
        sbt = self.clone()
        sbt.block_storage.add_(other)
        return sbt

    def add_(self, other):
        '''
        WARNING: Calling this function changes the block_storage, it has side effects if
        the block_storage referencing to an external tensor. This is the case when the 
        set_block_storage() function is called with clone=False. 

        other: must be a scalar or a Tensor or.
        '''
        if isinstance(other, ( int, float, torch.Tensor)):
            self.block_storage.add_( other )
        else:
            raise Exception(
                f'Currently only supports scalar or Tensor. type(other) = {type(other)}' )

        return self

    def __add__(self, other):
        '''
        other: must be a SparseBlockTensor object, or a tensor, or a scalar value.
        '''

        if isinstance(other, SparseBlockTensor):
            return self.add_sparse_block_matrix(other)
        elif isinstance(other, ( int, float, torch.Tensor)):
            # Not calling add_() to save a call to the isinstance() function.
            return self.add_broadcast(other)
        else:
            raise Exception(
                f'Currently only supports SparseBlockTensor, torch.Tensor, or scalar type. type(other) = {type(other)}' )

    def __radd__(self, other):
        '''
        other: must be a SparseBlockTensor object, or a tensor, or a scalar value.
        '''
        return self.__add__(other)

    def sub_broadcast(self, other):
        sbt = self.clone()
        sbt.block_storage.sub_(other)
        return sbt

    def sub_(self, other):
        '''
        WARNING: Calling this function changes the block_storage, it has side effects if
        the block_storage referencing to an external tensor. This is the case when the 
        set_block_storage() function is called with clone=False. 

        other: must be a scalar or a Tensor.
        '''
        if isinstance(other, ( int, float, torch.Tensor)):
            self.block_storage.sub_( other )
        else:
            raise Exception(
                f'Currently only supports scalar or Tensor. type(other) = {type(other)}' )

        return self

    def __sub__(self, other):
        '''
        other: must be a SparseBlockTensor object, or a tensor, or a scalar value.
        '''
        
        # This is not implemented as self.__add__( -1 * other) considering that
        # torch.Tensor.sub_() might be more efficient.

        if isinstance(other, SparseBlockTensor):
            return self.add_sparse_block_matrix( -1 * other )
        elif isinstance(other, ( int, float, torch.Tensor)):
            # Not calling sub_() to save a call to the isinstance() function.
            return self.sub_broadcast(other)
        else:
            raise Exception(
                f'Currently only supports SparseBlockTensor, torch.Tensor, or scalar type. type(other) = {type(other)}' )

    def __rsub__(self, other):
        '''
        other: must be a SparseBlockTensor object, or a tensor, or a scalar value.
        '''
        # This saves one multiplication compared with -1 * ( self - other )
        # When other is a SparseBlockTensor.
        return (-1 * self) + other

    def mul_(self, other):
        '''
        WARNING: Calling this function changes the block_storage, it has side effects if
        the block_storage referencing to an external tensor. This is the case when the 
        set_block_storage() function is called with clone=False. 

        other: must be a scalar or a Tensor.
        '''
        if isinstance(other, ( int, float, torch.Tensor )):
            self.block_storage.mul_(other)
        else:
            raise Exception(
                f'Currently only supports scalar type or Tensor. type(other) = {type(other)}' )

        return self

    def __mul__(self, other):
        '''
        Currently only supports multiplying by a scalar.
        '''
        res = self.clone()
        res.mul_(other)
        return res

    def __rmul__(self, other):
        '''
        Currently only supports multiplying by a scalar.
        '''
        return self.__mul__(other)

    def __matmul__(self, other):
        '''
        WARNING: This function causes the data to be offloaded to the CPU and back to the GPU if necessary.

        other: must be a SparseBlockTensor object.
        '''
        
        # ========== Checks. ==========

        assert ( self.col_block_structure.numel() == other.row_block_structure.numel() ), \
            f'Wrong dimension: \n>>> self: \n{self.dimension_str()}\n>>> other: \n{other.dimension_str()}'

        assert ( torch.equal( self.col_block_structure, other.row_block_structure ) ), \
            f'self and other has inconsistent col_block_structure and row_block_structure. '

        assert ( self.is_coalesced() and other.is_coalesced() ), \
            f'self.is_coalesced() = {self.is_coalesced()}, other.is_coalesced() = {other.is_coalesced()}'
        
        # ========== Checks done. ==========

        # ========== Off load the data from GPU to CPU and multiply. ==========

        bsr_self  = sbt_to_bsr_cpu(self)
        bsr_other = sbt_to_bsr_cpu(other)
        res_cpu   = bsr_self @ bsr_other

        # ========== Convert the result back to GPU. ==========
        res_gpu = bsr_cpu_to_sbt(res_cpu)

        return res_gpu
        
    def multiply_symmetric_upper_triangle(self, other):
        '''
        Not clear what this function does in g2o.
        '''
        raise NotImplementedError()

    def sym_permutate(slef, pinv: Iterable[int]):
        raise NotImplementedError()

    def fill_ccs(self):
        '''
        CCS - Column Compressed Structure.
        '''
        raise NotImplementedError()



