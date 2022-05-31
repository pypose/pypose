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

class DummyClass(object):
    def __init__(self, value, device=None) -> None:
        super().__init__()

        self.tensor = torch.Tensor([value]).to(device=device)

    def overwrite(self, value):
        # Create a temporary object.
        temp = DummyClass( value=value, device=self.tensor.device )
        
        # Overwrite the current instance's member.
        self.tensor = temp.tensor

def torch_lexsort(a, dim=-1):
    '''
    Found at https://discuss.pytorch.org/t/numpy-lexsort-equivalent-in-pytorch/47850/2
    '''
    assert dim == -1  # Transpose if you want differently
    assert a.ndim == 2  # Not sure what is numpy behaviour with > 2 dim

    sorted_values, inverse_indices = torch.unique(a, dim=dim, sorted=True, return_inverse=True)
    return sorted_values, inverse_indices

def sort_indices(indices: torch.Tensor):
    '''
    indices: 2D tensor.

    Returns:
    The sorted indices and the inverse indices.
    '''

    assert ( indices.ndim == 2 and indices.shape[0] == 2), f'indices.shape = {indices.shape}'

    return torch_lexsort(indices)

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

        #     30, 32, 34 = 0,  1,  2 + 30, 31, 32
        #     36, 38, 40   3,  4,  5   33, 34, 35

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

    def test_overwrite(self):
        print()
        show_delimeter('Test overwrite of a class member. ')

        test_entries = [
            { 'device': 'cpu'  },
            { 'device': 'cuda' }
        ]

        for entry in test_entries:
            print(f'entry = {entry}')

            device = entry['device']

            obj = DummyClass(1, device=device)
            obj.overwrite(2)

            true_tensor = torch.Tensor([2]).to(device=device)

            self.assertEqual( obj.tensor, true_tensor, f'test_overwrite fails with entry = f{entry}' )

    def test_torch_sparse_matrix(self):
        print()
        show_delimeter('Test PyTorch sparse matrix. ')

        shape = ( *Test_SparseBlockMatrix.shape_blocks, *Test_SparseBlockMatrix.block_shape )

        test_entries = [
            { 'device': 'cpu'  },
            { 'device': 'cuda' }
        ]

        for entry in test_entries:
            print(f'entry = {entry}')

            device = entry['device']

            indices = Test_SparseBlockMatrix.block_indices
            values = Test_SparseBlockMatrix.values_raw.detach().clone().to(device=device)

            scoo = torch.sparse_coo_tensor(indices, values, shape).to(device=device)
            values[0, 0, 0] = -1

            # This will make a copy of scoo
            scoo = scoo.coalesce()

            self.assertEqual(scoo.values()[0, 0, 0], -1, f'test_torch_sparse_matrix failed with entry = {entry}')

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

        shape = ( *Test_SparseBlockMatrix.shape_blocks, *Test_SparseBlockMatrix.block_shape )

        raw_true_scoo = torch.sparse_coo_tensor( 
           Test_SparseBlockMatrix.block_indices,
           Test_SparseBlockMatrix.values_raw,
           shape)

        test_entries = [
            { 'device': 'cpu'  },
            { 'device': 'cuda' }
        ]

        for entry in test_entries:
            print(entry)

            device = entry['device']
            values = Test_SparseBlockMatrix.values_raw.to(device)

            sbm = SparseBlockMatrix(Test_SparseBlockMatrix.block_shape, dtype=torch.float32, device=device)
            sbm.create(shape_blocks=Test_SparseBlockMatrix.shape_blocks, block_indices=Test_SparseBlockMatrix.block_indices)
            sbm.set_block_storage(values, clone=False)

            scoo = sbm_to_torch_sparse_coo(sbm)

            print(f'scoo.to_dense() = \n{scoo.to_dense()}')

            true_scoo = raw_true_scoo.to(device=device)

            try:
                torch_equal( scoo.to_dense(), true_scoo.to_dense() )
            except Exception as exc:
                print(exc)
                self.assertTrue(False, f'test_sbm_to_torch_sparse_coo failed with entry {entry}')

    def test_torch_sparse_coo_to_sbm(self):
        print()
        show_delimeter('torch_sparse_coo_to_sbm. ')

        shape = ( *Test_SparseBlockMatrix.shape_blocks, *Test_SparseBlockMatrix.block_shape )

        raw_true_scoo = torch.sparse_coo_tensor( 
           Test_SparseBlockMatrix.block_indices,
           Test_SparseBlockMatrix.values_raw,
           shape)

        raw_sbm = SparseBlockMatrix(Test_SparseBlockMatrix.block_shape, dtype=torch.float32)
        raw_sbm.create(shape_blocks=Test_SparseBlockMatrix.shape_blocks, block_indices=Test_SparseBlockMatrix.block_indices)
        raw_sbm.set_block_storage(Test_SparseBlockMatrix.values_raw, clone=False)

        test_entries = [
            { 'device': 'cpu'  },
            { 'device': 'cuda' }
        ]

        for entry in test_entries:
            print(entry)

            device = entry['device']

            true_scoo = raw_true_scoo.to(device=device)
            true_scoo = true_scoo.coalesce()

            sbm = raw_sbm.to(device=device)

            sbm_indices, inverse_indices = sort_indices(sbm.block_indices[:, :2].permute(1,0))

            try:
                torch_equal( sbm_indices, true_scoo.indices() )
            except Exception as exc:
                print(exc)
                self.assertTrue(False, f'test_torch_sparse_coo_to_sbm failed with entry {entry}')

            # rearange block_storage.
            print(inverse_indices)
            block_storage = torch.index_select(sbm.block_storage, 0, inverse_indices)

            try:
                torch_equal( block_storage, true_scoo.values() )
            except Exception as exc:
                print(exc)
                self.assertTrue(False, f'test_torch_sparse_coo_to_sbm failed with entry {entry}')

if __name__ == '__main__':
    import os
    print('Run %s. ' % (os.path.basename(__file__)))
    unittest.main()