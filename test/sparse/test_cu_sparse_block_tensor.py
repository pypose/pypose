# Unit test tools.
# from cgi import test
# import functools
# import inspect
import unittest

# System tools.
import copy
import numpy as np
import cupy as cp

# pypose.
# import pypose as pp
from pypose.sparse.sparse_block_tensor import (
    SparseBlockTensor, 
    sbt_to_bsr_cpu,
    sbt_to_cupy
)

# Test utils.
from .common import ( torch_equal, show_delimeter )

# PyTorch
import torch

class Test_cu_SparseBlockTensor(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        #      0,  1,  2,  3,  4,  5,  6,  7,  8
        # ======================================
        # 0 |  0,  1,  2,  x,  x,  x,  6,  7,  8
        # 1 |  3,  4,  5,  x,  x,  x,  9, 10, 11
        # 2 |  x,  x,  x, 12, 13, 14,  x,  x,  x
        # 3 |  x,  x,  x, 15, 16, 17,  x,  x,  x
        # 4 | 18, 19, 20,  x,  x,  x, 24, 25, 26
        # 5 | 21, 22, 23,  x,  x,  x, 27, 28, 29

        #     30, 32, 34 = 0,  1,  2 + 30, 31, 32
        #     36, 38, 40   3,  4,  5   33, 34, 35

        #      0,  1,  2,  3,  4,  5,  6,  7,  8
        # ======================================
        # 0 |  0,  1,  2,  x,  x,  x, 18, 19, 20
        # 1 |  3,  4,  5,  x,  x,  x, 21, 22, 23
        # 2 |  x,  x,  x, 12, 13, 14,  x,  x,  x
        # 3 |  x,  x,  x, 15, 16, 17,  x,  x,  x
        # 4 |  6,  7,  8,  x,  x,  x, 24, 25, 26
        # 5 |  9, 10, 11,  x,  x,  x, 27, 28, 29

        cls.block_shape = (2, 3)
        cls.layout = [
            [ 2, 6, 4, 2, 6 ],
            [ 3, 3, 6, 9, 9 ]
        ]
        cls.block_indices = [
            [ 0, 2, 1, 0, 2 ],
            [ 0, 0, 1, 2, 2 ]
        ]

        cls.bsr_indptr = np.array([ 0, 2, 3, 5 ])
        cls.bsr_indices = np.array([ 0, 2, 1, 0, 2 ])

        cls.rows = max(cls.layout[0])
        cls.cols = max(cls.layout[1])
        cls.shape = ( cls.rows, cls.cols )

        cls.rows_block = cls.rows // cls.block_shape[0]
        cls.cols_block = cls.cols // cls.block_shape[1]
        cls.shape_blocks = ( cls.rows_block, cls.cols_block )

        cls.rows_block_0 = 2
        cls.cols_block_0 = 3
        cls.rows_block_last = 2
        cls.cols_block_last = 3

        cls.row_base_block_0 = 0
        cls.col_base_block_0 = 0
        cls.row_base_block_last = 4
        cls.col_base_block_last = 6

        cls.last_block_row_idx = cls.rows_block - 1
        cls.last_block_col_idx = cls.cols_block - 1

        shape_values = ( len(cls.layout[0]), *cls.block_shape )
        cls.values_raw = torch.arange(30, dtype=torch.float32).view( shape_values )
        cls.bsr_data = torch.index_select( cls.values_raw, 0, torch.Tensor([0, 3, 2, 1, 4]).to(dtype=torch.int32) ).numpy()

    def test_sbt_to_cupy(self):
        print()
        show_delimeter('Test sbt to CuPy conversion. ')

        # Creawte the SparseblockMatrix.
        device = 'cuda'
        values = Test_cu_SparseBlockTensor.values_raw.to(device)
        sbt = SparseBlockTensor(Test_cu_SparseBlockTensor.block_shape, dtype=torch.float32, device=device)
        sbt.create(shape_blocks=Test_cu_SparseBlockTensor.shape_blocks, block_indices=Test_cu_SparseBlockTensor.block_indices)
        sbt.set_block_storage(values, clone=False)

        # Convert to CuPy.
        cu_sm = sbt_to_cupy(sbt)

        # Convert cu_sm to NumPy array.
        np_cu = cp.asnumpy( cu_sm.toarray() )

        print(f'np_cu = \n{np_cu}')

        # Convert sbt to SciPy BSR.
        bsr = sbt_to_bsr_cpu(sbt)

        # Convert bsr to NumPy array. 
        np_bsm = bsr.toarray()

        # Compare.
        assert np.allclose( np_cu, np_bsm ), f'test_sbt_to_cupy failed. '

    def test_sbt_to_cupy_duplicate_block_index(self):
        print()
        show_delimeter('Test sbt to CuPy conversion with duplicated block index. ')

        indices = copy.deepcopy( Test_cu_SparseBlockTensor.block_indices )
        indices[0][-1] = 0
        indices[1][-1] = 0

        # Creawte the SparseblockMatrix.
        device = 'cuda'
        values = Test_cu_SparseBlockTensor.values_raw.to(device)
        sbt = SparseBlockTensor(Test_cu_SparseBlockTensor.block_shape, dtype=torch.float32, device=device)
        sbt.create(shape_blocks=Test_cu_SparseBlockTensor.shape_blocks, block_indices=indices)
        sbt.set_block_storage(values, clone=False)

        # Convert to CuPy.
        cu_sm = sbt_to_cupy(sbt)

        # Convert cu_sm to NumPy array.
        np_cu = cp.asnumpy( cu_sm.toarray() )

        print(f'np_cu = \n{np_cu}')

        # Convert sbt to SciPy BSR.
        bsr = sbt_to_bsr_cpu(sbt)

        # Convert bsr to NumPy array. 
        np_bsm = bsr.toarray()

        # Compare.
        assert np.allclose( np_cu, np_bsm ), f'test_sbt_to_cupy_duplicate_block_index failed. '

if __name__ == '__main__':
    import os
    print('Run %s. ' % (os.path.basename(__file__)))
    unittest.main()