
from typing import List, Set, Dict, Tuple, Optional, Iterable, Union

import re
from numpy import block

import torch

# Globals.
INDEX_TYPE = torch.int64
FLOAT_TYPE = torch.float32

def sbm_to_torch_sparse_coo(sbm):
    s_shape = ( *sbm.shape_blocks, *sbm.block_shape )

    indices = sbm.block_indices[:, :2].permute((1, 0))

    return torch.sparse_coo_tensor( 
        indices, sbm.block_values, s_shape, dtype=sbm.dtype, device=sbm.device)

def torch_sparse_coo_to_sbm(s):
    assert ( s.is_sparse ), f's must be a sparse tensor. '
    assert ( s.layout == torch.sparse_coo ), f's must have the COO layout. s.layout = {s.layout}'
    assert ( s.ndim == 4 ), f's.shape == {s.shape}'

    if not s.is_coalesced():
        s = s.coalesce()
    
    shape_blocks = s.shape[:2]
    block_shape  = s.values.shape[1:3]

    sbm = SparseBlockMatrix( block_shape, dtype=s.dtype, device=s.device )
    sbm.create(shape_blocks=shape_blocks, device=s.device)
    sbm.block_indices[:, :2] = s.indices().detach().clone().permute((1, 0))
    sbm.block_storage = s.values().detach().clone()
    sbm.dtype = s.dtype

class SparseBlockMatrix(object):
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

    def create(self, 
        shape_blocks: Iterable[int],
        block_indices: Union[ List[List[int]], torch.Tensor ],
        device: Optional[str]=None) -> None:
        '''
        Argument definitions are copied from g2o:
        shape_blocks: number of blocks along vertical (number of block rows) and horizontal (number of block columns) directions.
        block_indices: 2D list or Tensor. 2 columns. Every row records the block index of a single block.
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

    @device.setter
    def device(self, d):
        self.row_block_structure = self.row_block_structure.to(device=d)
        self.col_block_structure = self.col_block_structure.to(device=d)
        
        if self.block_indices is not None:
            self.block_indices = self.block_indices.to(device=d)

        if self.block_storage is not None:
            self.block_storage = self.block_storage.to(device=d)

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
        new_matrix = SparseBlockMatrix( self.block_shape )
        new_matrix.col_block_structure = self.col_block_structure.clone()
        new_matrix.row_block_structure = self.row_block_structure.clone()
        
        if self.block_indices is not None:
            new_matrix.block_indices = self.block_indices.clone()

        if self.block_storage is not None:
            new_matrix.block_storage = self.block_storage.clone()

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
        A new SparseBlockMatrix object.
        '''
        raise NotImplementedError()

    def transpose_(self):
        '''
        In-place transpose.
        '''
        self.block_shape = self.block_shape[::-1]

        # We just need to swap row and column structures and the order of self.block_indices.
        self.row_block_structure, self.col_block_structure = self.col_block_structure, self.row_block_structure
        self.block_indices = torch.index_select( self.block_indices, 1, torch.LongTensor([1, 0, 2]) )

        # Transpose the blocks.
        self.block_storage = self.block_storage.permute((0, 2, 1))

    def transpose(self):
        '''
        Return a copy of the current sparse block matrix and transpose.
        '''
        # Make a copy of the current sparse block matrix.
        new_matrix = self.clone()
        return new_matrix.transpose_()

    def __add__(self, other):
        '''
        othet: must be a SparseBlockMatrix object.
        '''

        assert ( isinstance(other, SparseBlockMatrix) ), \
            f'Only supports adding two SparseBlockMatrix objects. type(other) = {type(other)}'

        assert ( self.rows == other.rows and self.cols == other.cols), \
            f'Incompatible dimensions: self: [{self.rows}, {self.cols}], other: [{other.rows}, {other.cols}]. '

        # Concatenate the raw data of the two SparseBlockMatrix.
        c_block_indices = torch.cat( ( self.block_indices, other.block_indices ), dim=0 )
        c_block_storage = torch.cat( ( self.block_storage, other.block_storage ), dim=0 )

        # Perform add by converting to a PyTorch sparse tensor.
        s = sbm_to_torch_sparse_coo( c_block_indices, c_block_storage, ( self.rows, self.cols ) )

        raise NotImplementedError()

    def __matmul__(self, other):
        '''
        other: must be a SparseBlockMatrix object or a scalar.
        '''
        raise NotImplementedError()

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

