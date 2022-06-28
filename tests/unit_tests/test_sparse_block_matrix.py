# Unit test tools.
from cgi import test
import functools
import inspect
import unittest

# System tools.
import numpy as np
from scipy.sparse import bsr_matrix

# pypose.
import pypose as pp
from pypose.optim.sparse_block_matrix import (
    SparseBlockMatrix, 
    sbm_to_torch_sparse_coo, torch_sparse_coo_to_sbm, 
    sbm_to_bsr_cpu, bsr_cpu_to_sbm)

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

        raw_true_sbm = SparseBlockMatrix(Test_SparseBlockMatrix.block_shape, dtype=torch.float32)
        raw_true_sbm.create(shape_blocks=Test_SparseBlockMatrix.shape_blocks, block_indices=Test_SparseBlockMatrix.block_indices)
        raw_true_sbm.set_block_storage(Test_SparseBlockMatrix.values_raw, clone=False)

        test_entries = [
            { 'device': 'cpu'  },
            { 'device': 'cuda' }
        ]

        for entry in test_entries:
            print(entry)

            device = entry['device']

            true_scoo = raw_true_scoo.to(device=device)
            true_scoo = true_scoo.coalesce()

            true_sbm = raw_true_sbm.to(device=device)

            true_sbm_indices, inverse_indices = sort_indices(true_sbm.block_indices[:, :2].permute(1,0))

            sbm = torch_sparse_coo_to_sbm(true_scoo)

            try:
                torch_equal( sbm.block_indices[:, :2], true_sbm_indices.permute((1, 0)) )
            except Exception as exc:
                print(exc)
                self.assertTrue(False, f'test_torch_sparse_coo_to_sbm failed with entry {entry}')

            # rearange block_storage.
            print(inverse_indices)
            true_block_storage = torch.index_select(true_sbm.block_storage, 0, inverse_indices)

            try:
                torch_equal( sbm.block_storage, true_block_storage )
            except Exception as exc:
                print(exc)
                self.assertTrue(False, f'test_torch_sparse_coo_to_sbm failed with entry {entry}')

    def test_add_sbm(self):
        print()
        show_delimeter('test add with another sbm. ')

        # The main sbm.
        raw_main_sbm = SparseBlockMatrix(Test_SparseBlockMatrix.block_shape, dtype=torch.float32)
        raw_main_sbm.create(shape_blocks=Test_SparseBlockMatrix.shape_blocks, block_indices=Test_SparseBlockMatrix.block_indices)
        raw_main_sbm.set_block_storage(Test_SparseBlockMatrix.values_raw, clone=False)

        # The other smb.
        other_block_storage = torch.rand_like(Test_SparseBlockMatrix.values_raw)
        raw_other_sbm = SparseBlockMatrix(Test_SparseBlockMatrix.block_shape, dtype=torch.float32)
        raw_other_sbm.create(shape_blocks=Test_SparseBlockMatrix.shape_blocks, block_indices=Test_SparseBlockMatrix.block_indices)
        raw_other_sbm.set_block_storage(other_block_storage, clone=False)

        # The result.
        raw_true_result_block_storage = raw_main_sbm.block_storage + raw_other_sbm.block_storage
        raw_result_sbm = SparseBlockMatrix(Test_SparseBlockMatrix.block_shape, dtype=torch.float32)
        raw_result_sbm.create(shape_blocks=Test_SparseBlockMatrix.shape_blocks, block_indices=Test_SparseBlockMatrix.block_indices)
        raw_result_sbm.set_block_storage(raw_true_result_block_storage, clone=False)
        raw_result_sbm = raw_result_sbm.coalesce()

        test_entries = [
            { 'device': 'cpu'  },
            { 'device': 'cuda' }
        ]

        for entry in test_entries:
            print(f'entry = {entry}')
            device = entry['device']

            main_sbm = raw_main_sbm.to(device=device)
            other_sbm = raw_other_sbm.to(device=device)
            true_result_sbm = raw_result_sbm.to(device=device)

            result = main_sbm + other_sbm
            result = result.coalesce()

            try:
                torch_equal( result.block_storage, true_result_sbm.block_storage )
            except Exception as exc:
                print(exc)
                self.assertTrue(False, f'test_add_sbm failed with entry {entry}')

    def test_add_sub_scalar(self):
        print()
        show_delimeter('test adding/subtracting a scalar. ')

        # The main sbm.
        raw_sbm = SparseBlockMatrix(Test_SparseBlockMatrix.block_shape, dtype=torch.float32)
        raw_sbm.create(shape_blocks=Test_SparseBlockMatrix.shape_blocks, block_indices=Test_SparseBlockMatrix.block_indices)
        raw_sbm.set_block_storage(Test_SparseBlockMatrix.values_raw, clone=False)

        test_entries = [
            { 'device': 'cpu',  'scalar': 1 },
            { 'device': 'cpu',  'scalar': 1.0 },
            { 'device': 'cpu',  'scalar': torch.Tensor([1]).to(dtype=torch.float32) },
            { 'device': 'cpu',  'scalar': torch.Tensor([1, 2, 3]).to(dtype=torch.float32).view((1, 3)) },
            { 'device': 'cpu',  'scalar': torch.Tensor([1, 2]).to(dtype=torch.float32).view((2, 1)) },
            { 'device': 'cuda', 'scalar': 1 },
            { 'device': 'cuda', 'scalar': 1.0 },
            { 'device': 'cuda', 'scalar': torch.Tensor([1]).to(dtype=torch.float32) },
            { 'device': 'cuda', 'scalar': torch.Tensor([1, 2, 3]).to(dtype=torch.float32).view((1, 3)) },
            { 'device': 'cuda', 'scalar': torch.Tensor([1, 2]).to(dtype=torch.float32).view((2, 1)) }
        ]

        for entry in test_entries:
            print(f'entry = {entry}')
            device = entry['device']

            # Transfer to the device.
            sbm = raw_sbm.to(device=device)
            scalar = entry['scalar']
            if isinstance(scalar, (int, float)):
                other = scalar
            else:
                other = scalar.to(device=device)
            block_storage = Test_SparseBlockMatrix.values_raw.to(device=device)

            # ========== Addition. ==========

            # The result values.
            true_result_values = block_storage + other

            # Perform the addition from left.
            result = sbm + other

            try:
                torch_equal( result.block_storage, true_result_values )
            except Exception as exc:
                print(exc)
                self.assertTrue(False, f'test_add_sub_scalar (add left) failed with entry {entry}')

            # Perform the addition from right.
            result = other + sbm

            try:
                torch_equal( result.block_storage, true_result_values )
            except Exception as exc:
                print(exc)
                self.assertTrue(False, f'test_add_sub_scalar (add right) failed with entry {entry}')

            # ========== Substraction. ==========

            # Perform the subtraction from left.
            true_result_values = block_storage - other
            result = sbm - other

            try:
                torch_equal( result.block_storage, true_result_values )
            except Exception as exc:
                print(exc)
                self.assertTrue(False, f'test_add_sub_scalar (sub left) failed with entry {entry}')

            # Perform the subtraction from right.
            true_result_values = other - block_storage
            result = other - sbm

            try:
                torch_equal( result.block_storage, true_result_values )
            except Exception as exc:
                print(exc)
                self.assertTrue(False, f'test_add_sub_scalar (sub right) failed with entry {entry}')

    def test_add_scalar_inplace(self):
        print()
        show_delimeter('test inplace-adding a scalar. ')

        # The main sbm.
        raw_sbm = SparseBlockMatrix(Test_SparseBlockMatrix.block_shape, dtype=torch.float32)
        raw_sbm.create(shape_blocks=Test_SparseBlockMatrix.shape_blocks, block_indices=Test_SparseBlockMatrix.block_indices)
        raw_sbm.set_block_storage(Test_SparseBlockMatrix.values_raw, clone=False)

        test_entries = [
            { 'device': 'cpu',  'scalar': 1 },
            { 'device': 'cpu',  'scalar': 1.0 },
            { 'device': 'cpu',  'scalar': torch.Tensor([1]).to(dtype=torch.float32) },
            { 'device': 'cpu',  'scalar': torch.Tensor([1, 2, 3]).to(dtype=torch.float32).view((1, 3)) },
            { 'device': 'cpu',  'scalar': torch.Tensor([1, 2]).to(dtype=torch.float32).view((2, 1)) },
            { 'device': 'cuda', 'scalar': 1 },
            { 'device': 'cuda', 'scalar': 1.0 },
            { 'device': 'cuda', 'scalar': torch.Tensor([1]).to(dtype=torch.float32) },
            { 'device': 'cuda', 'scalar': torch.Tensor([1, 2, 3]).to(dtype=torch.float32).view((1, 3)) },
            { 'device': 'cuda', 'scalar': torch.Tensor([1, 2]).to(dtype=torch.float32).view((2, 1)) }
        ]

        for entry in test_entries:
            print(f'entry = {entry}')
            device = entry['device']

            # Transfer to the device.
            sbm = raw_sbm.clone().to(device=device)
            scalar = entry['scalar']
            if isinstance(scalar, (int, float)):
                other = scalar
            else:
                other = scalar.to(device=device)
            block_storage = Test_SparseBlockMatrix.values_raw.to(device=device)

            # The result values.
            true_result_values = block_storage + other

            # Perform the addition.
            sbm.add_(other)

            try:
                torch_equal( sbm.block_storage, true_result_values )
            except Exception as exc:
                print(exc)
                self.assertTrue(False, f'test_add_scalar_inplace failed with entry {entry}')

    def test_sbm_2_bsr_cpu(self):
        print()
        show_delimeter('Test sbm to bsr conversion. ')

        raw_true_bsr = bsr_matrix(
            ( Test_SparseBlockMatrix.bsr_data, 
              Test_SparseBlockMatrix.bsr_indices,
              Test_SparseBlockMatrix.bsr_indptr ),
            shape=Test_SparseBlockMatrix.shape ).toarray()

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

            bsr = sbm_to_bsr_cpu(sbm)
            bsr = bsr.toarray()

            self.assertTrue( np.allclose( bsr, raw_true_bsr ), f'test_sbm_2_bsr_cpu failed with entry {entry}' )

    def test_bsr_cpu_2_sbm(self):
        print()
        show_delimeter('Test bsr to sbm conversion. ')

        # Create the BSR matrix.
        bsr = bsr_matrix( 
            ( Test_SparseBlockMatrix.bsr_data, 
              Test_SparseBlockMatrix.bsr_indices, 
              Test_SparseBlockMatrix.bsr_indptr ), 
            shape=Test_SparseBlockMatrix.shape )
        
        # Convert BSR to SBM.
        sbm = bsr_cpu_to_sbm(bsr)

        # Check equality.
        dense_sbm = sbm_to_torch_sparse_coo(sbm).to_dense().permute((0,2,1,3)).numpy().reshape(
            ( Test_SparseBlockMatrix.shape_blocks[0]*Test_SparseBlockMatrix.block_shape[0], 
              Test_SparseBlockMatrix.shape_blocks[1]*Test_SparseBlockMatrix.block_shape[1] ) )
        dense_bsr = bsr.toarray()

        print(f'dense_sbm = \n{dense_sbm}')
        print(f'dense_bsr = \n{dense_bsr}')

        self.assertTrue( np.allclose( dense_sbm, dense_bsr ), f'test_bsr_cpu_2_sbm failed' )

    def test_matmul(self):
        print()
        show_delimeter('Test matmul. ')

        # The Sparse Block Matrix.
        raw_sbm = SparseBlockMatrix(Test_SparseBlockMatrix.block_shape, dtype=torch.float32)
        raw_sbm.create(shape_blocks=Test_SparseBlockMatrix.shape_blocks, block_indices=Test_SparseBlockMatrix.block_indices)
        raw_sbm.set_block_storage(Test_SparseBlockMatrix.values_raw, clone=False)

        raw_sbm = raw_sbm.coalesce()

        # The equavelent dense NumPy array.
        dense_array = sbm_to_bsr_cpu(raw_sbm).toarray()

        # The matmul results.
        true_res_a_at = dense_array @ dense_array.transpose()
        true_res_at_a = dense_array.transpose() @ dense_array

        # Show the true values.
        print(f'true_res_at_a = \n{true_res_at_a}')
        print(f'true_res_a_at = \n{true_res_a_at}')

        test_entries = [
            { 'device': 'cpu'  },
            { 'device': 'cuda' }
        ]

        for entry in test_entries:
            print(entry)

            device = entry['device']

            # Transfer the Sparse Block Matrix to the device.
            sbm = raw_sbm.to(device=device)

            # a.T @ a.
            res_at_a = sbm.transpose().coalesce() @ sbm
            res_at_a_cpu = sbm_to_bsr_cpu(res_at_a).toarray()
            self.assertTrue( np.allclose( res_at_a_cpu, true_res_at_a ) )

            # a @ a.T.
            res_a_at = sbm @ sbm.transpose().coalesce()
            res_a_at_cpu = sbm_to_bsr_cpu(res_a_at).toarray()
            self.assertTrue( np.allclose( res_a_at_cpu, true_res_a_at ) )

    def test_multiply_scalar(self):
        print()
        show_delimeter('test multiplying a scalar. ')

        # The main sbm.
        raw_sbm = SparseBlockMatrix(Test_SparseBlockMatrix.block_shape, dtype=torch.float32)
        raw_sbm.create(shape_blocks=Test_SparseBlockMatrix.shape_blocks, block_indices=Test_SparseBlockMatrix.block_indices)
        raw_sbm.set_block_storage(Test_SparseBlockMatrix.values_raw, clone=False)

        test_entries = [
            { 'device': 'cpu',  'scalar': 2 },
            { 'device': 'cpu',  'scalar': 2.0 },
            { 'device': 'cpu',  'scalar': torch.Tensor([2]).to(dtype=torch.float32) },
            { 'device': 'cpu',  'scalar': torch.Tensor([1, 2, 3]).to(dtype=torch.float32).view((1, 3)) },
            { 'device': 'cpu',  'scalar': torch.Tensor([1, 2]).to(dtype=torch.float32).view((2, 1)) },
            { 'device': 'cuda', 'scalar': 2 },
            { 'device': 'cuda', 'scalar': 2.0 },
            { 'device': 'cuda', 'scalar': torch.Tensor([2]).to(dtype=torch.float32) },
            { 'device': 'cuda', 'scalar': torch.Tensor([1, 2, 3]).to(dtype=torch.float32).view((1, 3)) },
            { 'device': 'cuda', 'scalar': torch.Tensor([1, 2]).to(dtype=torch.float32).view((2, 1)) }
        ]

        for entry in test_entries:
            print(f'entry = {entry}')
            device = entry['device']

            # Transfer to the device.
            sbm = raw_sbm.to(device=device)
            scalar = entry['scalar']
            if isinstance(scalar, (int, float)):
                other = scalar
            else:
                other = scalar.to(device=device)
            block_storage = Test_SparseBlockMatrix.values_raw.to(device=device)

            # The result values.
            true_result_values = block_storage * other

            # Perform the multiplication from left.
            result = sbm * other

            try:
                torch_equal( result.block_storage, true_result_values )
            except Exception as exc:
                print(exc)
                self.assertTrue(False, f'test_multiply_scalar (left) failed with entry {entry}')

            # Perform the multiplication from right.
            result = other * sbm

            try:
                torch_equal( result.block_storage, true_result_values )
            except Exception as exc:
                print(exc)
                self.assertTrue(False, f'test_multiply_scalar (right) failed with entry {entry}')

    def test_multiply_scalar_inplace(self):
        print()
        show_delimeter('test inplace multiplying a scalar. ')

        # The main sbm.
        raw_sbm = SparseBlockMatrix(Test_SparseBlockMatrix.block_shape, dtype=torch.float32)
        raw_sbm.create(shape_blocks=Test_SparseBlockMatrix.shape_blocks, block_indices=Test_SparseBlockMatrix.block_indices)
        raw_sbm.set_block_storage(Test_SparseBlockMatrix.values_raw, clone=False)

        test_entries = [
            { 'device': 'cpu',  'scalar': 2 },
            { 'device': 'cpu',  'scalar': 2.0 },
            { 'device': 'cpu',  'scalar': torch.Tensor([2]).to(dtype=torch.float32) },
            { 'device': 'cpu',  'scalar': torch.Tensor([1, 2, 3]).to(dtype=torch.float32).view((1, 3)) },
            { 'device': 'cpu',  'scalar': torch.Tensor([1, 2]).to(dtype=torch.float32).view((2, 1)) },
            { 'device': 'cuda', 'scalar': 2 },
            { 'device': 'cuda', 'scalar': 2.0 },
            { 'device': 'cuda', 'scalar': torch.Tensor([2]).to(dtype=torch.float32) },
            { 'device': 'cuda', 'scalar': torch.Tensor([1, 2, 3]).to(dtype=torch.float32).view((1, 3)) },
            { 'device': 'cuda', 'scalar': torch.Tensor([1, 2]).to(dtype=torch.float32).view((2, 1)) }
        ]

        for entry in test_entries:
            print(f'entry = {entry}')
            device = entry['device']

            # Transfer to the device.
            sbm = raw_sbm.clone().to(device=device)
            scalar = entry['scalar']
            if isinstance(scalar, (int, float)):
                other = scalar
            else:
                other = scalar.to(device=device)
            block_storage = Test_SparseBlockMatrix.values_raw.to(device=device)

            # The result values.
            true_result_values = block_storage * other

            # Perform the multiplication from left.
            sbm.mul_(other)

            try:
                torch_equal( sbm.block_storage, true_result_values )
            except Exception as exc:
                print(exc)
                self.assertTrue(False, f'test_inplace_multiply_scalar failed with entry {entry}')

if __name__ == '__main__':
    import os
    print('Run %s. ' % (os.path.basename(__file__)))
    unittest.main()