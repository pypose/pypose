# Unit test tools.
import functools
import inspect
import unittest

# System tools.
import numpy as np

# pypose.
import pypose as pp
from pypose.optim.sparse_block_matrix import (
    SparseBlockMatrix, sbm_to_torch_sparse_coo, torch_sparse_coo_to_sbm)

# Test utils.
from tests.unit_tests.common import ( torch_equal, show_delimeter )

# PyTorch
import torch

class Test_SparseBlockMatrix(unittest.TestCase):

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

        cls.block_shape = (2, 3)
        cls.layout = [
            [ 2, 2, 4, 6, 6 ],
            [ 3, 9, 6, 3, 9 ]
        ]
        cls.block_indices = [
            [ 0, 2, 1, 0, 2 ],
            [ 0, 0, 1, 2, 2 ]
        ]

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

    def test_creation_empty(self):
        print()
        show_delimeter('Test SBM empty creation. ')

        test_entries = [
            { 'device': 'cpu', 'torch_device': torch.device('cpu') },
            { 'device': 'cuda', 'torch_device': torch.device('cuda:0') } # Naive test with the first GPU.
        ]

        for entry in test_entries:
            print(f'entry = {entry}')

            device = entry['device']

            sbm = SparseBlockMatrix((2, 2), dtype=torch.float32, device=device)

            print(f'sbm.rows = {sbm.rows}')
            print(f'sbm.cols = {sbm.cols}')
            print(f'sbm.shape = {sbm.shape}')
            print(f'sbm.shape_blocks = {sbm.shape_blocks}')
            print(f'sbm.device = {sbm.device}')
            print(f'sbm.dtype = {sbm.dtype}')
            print(f'sbm.type() = {sbm.type()}')

            self.assertEqual(sbm.rows, 0, f'sbm.rows should be 0')
            self.assertEqual(sbm.cols, 0, f'sbm.cols should be 0')
            self.assertEqual(sbm.shape, (0, 0), f'sbm.shape should be (0, 0)')
            self.assertEqual(sbm.shape_blocks, (0, 0), f'sbm.shape_blocks should be (0, 0)')
            self.assertEqual(sbm.device, entry['torch_device'], f'sbm.device should be cuda:0')
            self.assertEqual(sbm.dtype, torch.float32, f'sbm.dtype should be torch.float32')
            self.assertEqual(sbm.type(), torch.float32, f'sbm.type() should return torch.float32')

    def test_creation(self):
        print()
        show_delimeter('Test SBM creation. ')

        test_entries = [
            { 'device': 'cpu', 'torch_device': torch.device('cpu') },
            { 'device': 'cuda', 'torch_device': torch.device('cuda:0') } # Naive test with the first GPU.
        ]

        for entry in test_entries:
            print(f'entry = {entry}')

            device = entry['device']
            values = Test_SparseBlockMatrix.values_raw.to(device)

            sbm = SparseBlockMatrix(Test_SparseBlockMatrix.block_shape, dtype=torch.float32, device=device)
            sbm.create(shape_blocks=Test_SparseBlockMatrix.shape_blocks, block_indices=Test_SparseBlockMatrix.block_indices)

            print(f'sbm.rows = {sbm.rows}')
            print(f'sbm.cols = {sbm.cols}')
            print(f'sbm.shape = {sbm.shape}')
            print(f'sbm.shape_blocks = {sbm.shape_blocks}')
            print(f'sbm.device = {sbm.device}')
            print(f'sbm.dtype = {sbm.dtype}')
            print(f'sbm.type() = {sbm.type()}')

            self.assertEqual(sbm.rows, Test_SparseBlockMatrix.rows, f'sbm.rows should be {Test_SparseBlockMatrix.rows}')
            self.assertEqual(sbm.cols, Test_SparseBlockMatrix.cols, f'sbm.cols should be {Test_SparseBlockMatrix.cols}')
            self.assertEqual(sbm.shape, Test_SparseBlockMatrix.shape, f'sbm.shape should be {Test_SparseBlockMatrix.shape}')
            self.assertEqual(sbm.shape_blocks, Test_SparseBlockMatrix.shape_blocks, f'sbm.shape_blocks should be {Test_SparseBlockMatrix.shape_blocks}')
            self.assertEqual(sbm.device, entry['torch_device'], f'sbm.device should be {entry["torch_device"]}')
            self.assertEqual(sbm.dtype, torch.float32, f'sbm.dtype should be torch.float32')
            self.assertEqual(sbm.type(), torch.float32, f'sbm.type() should return torch.float32')

            sbm.set_block_storage(values)
            self.assertEqual(sbm.type(), torch.float32, f'sbm.type() should return torch.float32')

            self.assertEqual( sbm.rows_of_block(0), Test_SparseBlockMatrix.rows_block_0, f'sbm.rows_of_block(0) = {sbm.rows_of_block(0)}, not {Test_SparseBlockMatrix.rows_block_0}' )
            self.assertEqual( sbm.cols_of_block(0), Test_SparseBlockMatrix.cols_block_0, f'sbm.cols_of_block(0) = {sbm.cols_of_block(0)}, not {Test_SparseBlockMatrix.cols_block_0}' )

            self.assertEqual( 
                sbm.rows_of_block(Test_SparseBlockMatrix.last_block_row_idx), 
                Test_SparseBlockMatrix.rows_block_last, 
                f'sbm.rows_of_block({Test_SparseBlockMatrix.last_block_row_idx}) = {sbm.rows_of_block(Test_SparseBlockMatrix.last_block_row_idx)}, not {Test_SparseBlockMatrix.rows_block_last}' )
            self.assertEqual( 
                sbm.cols_of_block(Test_SparseBlockMatrix.last_block_col_idx), 
                Test_SparseBlockMatrix.cols_block_last, 
                f'sbm.cols_of_block({Test_SparseBlockMatrix.last_block_col_idx}) = {sbm.cols_of_block(Test_SparseBlockMatrix.last_block_col_idx)}, not {Test_SparseBlockMatrix.cols_block_last}' )

    def test_to(self):
        print()
        show_delimeter('Test SBM to(). ')

        test_entries = [
            { 'device': 'cpu',  'torch_device': torch.device('cpu')   },
            { 'device': 'cuda', 'torch_device': torch.device('cuda:0')} # Naive test with the first GPU.
        ]

        # Test sequence gonna be
        loop_args = [
            {'device': 'cpu',  'dtype': torch.float32, 'copy': False},
            {'device': 'cuda', 'dtype': torch.float32, 'copy': False},
            {'device': 'cpu',  'dtype': torch.int,     'copy': False},
            {'device': 'cuda', 'dtype': torch.int,     'copy': False},
            {'device': 'cpu',  'dtype': torch.float32, 'copy': True},
            {'device': 'cuda', 'dtype': torch.float32, 'copy': True},
            {'device': 'cpu',  'dtype': torch.int,     'copy': True},
            {'device': 'cuda', 'dtype': torch.int,     'copy': True}
        ]

        for entry in test_entries:
            print(f'entry = {entry}')

            device = entry['device']
            values = Test_SparseBlockMatrix.values_raw.to(device)

            sbm = SparseBlockMatrix(Test_SparseBlockMatrix.block_shape, dtype=torch.float32, device=device)
            sbm.create(shape_blocks=Test_SparseBlockMatrix.shape_blocks, block_indices=Test_SparseBlockMatrix.block_indices)
            sbm.set_block_storage(values, clone=False)

            if device == 'cpu':
                copied = [ False, True, True, True, True, True, True, True ]
            else:
                copied = [ True, False, True, True, True, True, True, True ]

            for args, flag_copy in zip(loop_args, copied):
                print(f'args = {args}')
                sbm_new = sbm.to( **args )

                self.assertEqual(sbm_new.dtype, args['dtype'], f'sbm_new.dtype should be {args["dtype"]}')
                self.assertEqual( 
                    id(sbm_new) == id(sbm),
                    not flag_copy,
                    f'Copy should not happen with args = {args}' )

    def test_sbm_to_torch_sparse_coo(self):
        print()
        show_delimeter('Test sbm_to_torch_sparse_coo(). ')

        test_entries = [
            { 'device': 'cpu'  },
            { 'device': 'cuda' }
        ]

        for entry in test_entries:
            print(entry)

if __name__ == '__main__':
    import os
    print('Run %s. ' % (os.path.basename(__file__)))
    unittest.main()