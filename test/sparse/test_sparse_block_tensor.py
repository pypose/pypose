# Unit test tools.sprase_bloc_senor
from cgi import test
import functools
import inspect
import unittest

# System tools.
import numpy as np
from scipy.sparse import bsr_matrix

# pypose.
import pypose as pp
from pypose.sparse.sparse_block_tensor import (
    SparseBlockTensor, 
    sbt_to_torch_sparse_coo, torch_sparse_coo_to_sbt, 
    sbt_to_bsr_cpu, bsr_cpu_to_sbt)

# Test utils.
from .common import ( torch_equal, show_delimeter )

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

class TestSparseBlockTensor(unittest.TestCase):

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

            assert obj.tensor == true_tensor, f'test_overwrite fails with entry = f{entry}'

    def test_torch_sparse_matrix(self):
        print()
        show_delimeter('Test PyTorch sparse matrix. ')

        shape = ( *TestSparseBlockTensor.shape_blocks, *TestSparseBlockTensor.block_shape )

        test_entries = [
            { 'device': 'cpu'  },
            { 'device': 'cuda' }
        ]

        for entry in test_entries:
            print(f'entry = {entry}')

            device = entry['device']

            indices = TestSparseBlockTensor.block_indices
            values = TestSparseBlockTensor.values_raw.detach().clone().to(device=device)

            scoo = torch.sparse_coo_tensor(indices, values, shape).to(device=device)
            values[0, 0, 0] = -1

            # This will make a copy of scoo
            scoo = scoo.coalesce()

            assert scoo.values()[0, 0, 0] == -1, f'test_torch_sparse_matrix failed with entry = {entry}'

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

            sbt = SparseBlockTensor((2, 2), dtype=torch.float32, device=device)

            print(f'sbt.rows = {sbt.rows}')
            print(f'sbt.cols = {sbt.cols}')
            print(f'sbt.shape = {sbt.shape}')
            print(f'sbt.shape_blocks = {sbt.shape_blocks}')
            print(f'sbt.device = {sbt.device}')
            print(f'sbt.dtype = {sbt.dtype}')
            print(f'sbt.type() = {sbt.type()}')

            assert sbt.rows == 0, f'sbt.rows should be 0'
            assert sbt.cols == 0, f'sbt.cols should be 0'
            assert sbt.shape == (0, 0), f'sbt.shape should be (0, 0)'
            assert sbt.shape_blocks == (0, 0), f'sbt.shape_blocks should be (0, 0)'
            assert sbt.device == entry['torch_device'], f'sbt.device should be cuda:0'
            assert sbt.dtype == torch.float32, f'sbt.dtype should be torch.float32'
            assert sbt.type() == torch.float32, f'sbt.type() should return torch.float32'

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
            values = TestSparseBlockTensor.values_raw.to(device)

            sbt = SparseBlockTensor(TestSparseBlockTensor.block_shape, dtype=torch.float32, device=device)
            sbt.create(shape_blocks=TestSparseBlockTensor.shape_blocks, block_indices=TestSparseBlockTensor.block_indices)

            print(f'sbt.rows = {sbt.rows}')
            print(f'sbt.cols = {sbt.cols}')
            print(f'sbt.shape = {sbt.shape}')
            print(f'sbt.shape_blocks = {sbt.shape_blocks}')
            print(f'sbt.device = {sbt.device}')
            print(f'sbt.dtype = {sbt.dtype}')
            print(f'sbt.type() = {sbt.type()}')

            assert sbt.rows == TestSparseBlockTensor.rows, f'sbt.rows should be {TestSparseBlockTensor.rows}'
            assert sbt.cols == TestSparseBlockTensor.cols, f'sbt.cols should be {TestSparseBlockTensor.cols}'
            assert sbt.shape == TestSparseBlockTensor.shape, f'sbt.shape should be {TestSparseBlockTensor.shape}'
            assert sbt.shape_blocks == TestSparseBlockTensor.shape_blocks, f'sbt.shape_blocks should be {TestSparseBlockTensor.shape_blocks}'
            assert sbt.device == entry['torch_device'], f'sbt.device should be {entry["torch_device"]}'
            assert sbt.dtype == torch.float32, f'sbt.dtype should be torch.float32'
            assert sbt.type() == torch.float32, f'sbt.type() should return torch.float32'

            sbt.set_block_storage(values)
            assert sbt.type() == torch.float32, f'sbt.type() should return torch.float32'

            assert sbt.rows_of_block(0) == TestSparseBlockTensor.rows_block_0, f'sbt.rows_of_block(0) = {sbt.rows_of_block(0)}, not {TestSparseBlockTensor.rows_block_0}'
            assert sbt.cols_of_block(0) == TestSparseBlockTensor.cols_block_0, f'sbt.cols_of_block(0) = {sbt.cols_of_block(0)}, not {TestSparseBlockTensor.cols_block_0}'

            assert sbt.rows_of_block(TestSparseBlockTensor.last_block_row_idx) == \
                TestSparseBlockTensor.rows_block_last, \
                f'sbt.rows_of_block({TestSparseBlockTensor.last_block_row_idx}) = {sbt.rows_of_block(TestSparseBlockTensor.last_block_row_idx)}, not {TestSparseBlockTensor.rows_block_last}'
            assert sbt.cols_of_block(TestSparseBlockTensor.last_block_col_idx) == \
                TestSparseBlockTensor.cols_block_last, \
                f'sbt.cols_of_block({TestSparseBlockTensor.last_block_col_idx}) = {sbt.cols_of_block(TestSparseBlockTensor.last_block_col_idx)}, not {TestSparseBlockTensor.cols_block_last}'

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
            values = TestSparseBlockTensor.values_raw.to(device)

            sbt = SparseBlockTensor(TestSparseBlockTensor.block_shape, dtype=torch.float32, device=device)
            sbt.create(shape_blocks=TestSparseBlockTensor.shape_blocks, block_indices=TestSparseBlockTensor.block_indices)
            sbt.set_block_storage(values, clone=False)

            if device == 'cpu':
                copied = [ False, True, True, True, True, True, True, True ]
            else:
                copied = [ True, False, True, True, True, True, True, True ]

            for args, flag_copy in zip(loop_args, copied):
                print(f'args = {args}')
                sbt_new = sbt.to( **args )

                assert sbt_new.dtype == args['dtype'], f'sbt_new.dtype should be {args["dtype"]}'
                assert (id(sbt_new) == id(sbt)) == \
                    (not flag_copy), \
                    f'Copy should not happen with args = {args}'

    def test_sbt_to_torch_sparse_coo(self):
        print()
        show_delimeter('Test sbt_to_torch_sparse_coo(). ')

        shape = ( *TestSparseBlockTensor.shape_blocks, *TestSparseBlockTensor.block_shape )

        raw_true_scoo = torch.sparse_coo_tensor( 
           TestSparseBlockTensor.block_indices,
           TestSparseBlockTensor.values_raw,
           shape)

        test_entries = [
            { 'device': 'cpu'  },
            { 'device': 'cuda' }
        ]

        for entry in test_entries:
            print(entry)

            device = entry['device']
            values = TestSparseBlockTensor.values_raw.to(device)

            sbt = SparseBlockTensor(TestSparseBlockTensor.block_shape, dtype=torch.float32, device=device)
            sbt.create(shape_blocks=TestSparseBlockTensor.shape_blocks, block_indices=TestSparseBlockTensor.block_indices)
            sbt.set_block_storage(values, clone=False)

            scoo = sbt_to_torch_sparse_coo(sbt)

            print(f'scoo.to_dense() = \n{scoo.to_dense()}')

            true_scoo = raw_true_scoo.to(device=device)

            try:
                torch_equal( scoo.to_dense(), true_scoo.to_dense() )
            except Exception as exc:
                print(exc)
                assert False, f'test_sbt_to_torch_sparse_coo failed with entry {entry}'

    def test_torch_sparse_coo_to_sbt(self):
        print()
        show_delimeter('torch_sparse_coo_to_sbt. ')

        shape = ( *TestSparseBlockTensor.shape_blocks, *TestSparseBlockTensor.block_shape )

        raw_true_scoo = torch.sparse_coo_tensor( 
           TestSparseBlockTensor.block_indices,
           TestSparseBlockTensor.values_raw,
           shape)

        raw_true_sbt = SparseBlockTensor(TestSparseBlockTensor.block_shape, dtype=torch.float32)
        raw_true_sbt.create(shape_blocks=TestSparseBlockTensor.shape_blocks, block_indices=TestSparseBlockTensor.block_indices)
        raw_true_sbt.set_block_storage(TestSparseBlockTensor.values_raw, clone=False)

        test_entries = [
            { 'device': 'cpu'  },
            { 'device': 'cuda' }
        ]

        for entry in test_entries:
            print(entry)

            device = entry['device']

            true_scoo = raw_true_scoo.to(device=device)
            true_scoo = true_scoo.coalesce()

            true_sbt = raw_true_sbt.to(device=device)

            true_sbt_indices, inverse_indices = sort_indices(true_sbt.block_indices[:, :2].permute(1,0))

            sbt = torch_sparse_coo_to_sbt(true_scoo)

            try:
                torch_equal( sbt.block_indices[:, :2], true_sbt_indices.permute((1, 0)) )
            except Exception as exc:
                print(exc)
                assert False, f'test_torch_sparse_coo_to_sbt failed with entry {entry}'

            # rearange block_storage.
            print(inverse_indices)
            true_block_storage = torch.index_select(true_sbt.block_storage, 0, inverse_indices)

            try:
                torch_equal( sbt.block_storage, true_block_storage )
            except Exception as exc:
                print(exc)
                assert False, f'test_torch_sparse_coo_to_sbt failed with entry {entry}'

    def test_add_sbt(self):
        print()
        show_delimeter('test add with another sbt. ')

        # The main sbt.
        raw_main_sbt = SparseBlockTensor(TestSparseBlockTensor.block_shape, dtype=torch.float32)
        raw_main_sbt.create(shape_blocks=TestSparseBlockTensor.shape_blocks, block_indices=TestSparseBlockTensor.block_indices)
        raw_main_sbt.set_block_storage(TestSparseBlockTensor.values_raw, clone=False)

        # The other smb.
        other_block_storage = torch.rand_like(TestSparseBlockTensor.values_raw)
        raw_other_sbt = SparseBlockTensor(TestSparseBlockTensor.block_shape, dtype=torch.float32)
        raw_other_sbt.create(shape_blocks=TestSparseBlockTensor.shape_blocks, block_indices=TestSparseBlockTensor.block_indices)
        raw_other_sbt.set_block_storage(other_block_storage, clone=False)

        # The result.
        raw_true_result_block_storage = raw_main_sbt.block_storage + raw_other_sbt.block_storage
        raw_result_sbt = SparseBlockTensor(TestSparseBlockTensor.block_shape, dtype=torch.float32)
        raw_result_sbt.create(shape_blocks=TestSparseBlockTensor.shape_blocks, block_indices=TestSparseBlockTensor.block_indices)
        raw_result_sbt.set_block_storage(raw_true_result_block_storage, clone=False)
        raw_result_sbt = raw_result_sbt.coalesce()

        test_entries = [
            { 'device': 'cpu'  },
            { 'device': 'cuda' }
        ]

        for entry in test_entries:
            print(f'entry = {entry}')
            device = entry['device']

            main_sbt = raw_main_sbt.to(device=device)
            other_sbt = raw_other_sbt.to(device=device)
            true_result_sbt = raw_result_sbt.to(device=device)

            result = main_sbt + other_sbt
            result = result.coalesce()

            try:
                torch_equal( result.block_storage, true_result_sbt.block_storage )
            except Exception as exc:
                print(exc)
                assert False, f'test_add_sbt failed with entry {entry}'

    def test_add_sub_scalar(self):
        print()
        show_delimeter('test adding/subtracting a scalar. ')

        # The main sbt.
        raw_sbt = SparseBlockTensor(TestSparseBlockTensor.block_shape, dtype=torch.float32)
        raw_sbt.create(shape_blocks=TestSparseBlockTensor.shape_blocks, block_indices=TestSparseBlockTensor.block_indices)
        raw_sbt.set_block_storage(TestSparseBlockTensor.values_raw, clone=False)

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
            sbt = raw_sbt.to(device=device)
            scalar = entry['scalar']
            if isinstance(scalar, (int, float)):
                other = scalar
            else:
                other = scalar.to(device=device)
            block_storage = TestSparseBlockTensor.values_raw.to(device=device)

            # ========== Addition. ==========

            # The result values.
            true_result_values = block_storage + other

            # Perform the addition from left.
            result = sbt + other

            try:
                torch_equal( result.block_storage, true_result_values )
            except Exception as exc:
                print(exc)
                assert False, f'test_add_sub_scalar (add left) failed with entry {entry}'

            # Perform the addition from right.
            result = other + sbt

            try:
                torch_equal( result.block_storage, true_result_values )
            except Exception as exc:
                print(exc)
                assert False, f'test_add_sub_scalar (add right) failed with entry {entry}'

            # ========== Substraction. ==========

            # Perform the subtraction from left.
            true_result_values = block_storage - other
            result = sbt - other

            try:
                torch_equal( result.block_storage, true_result_values )
            except Exception as exc:
                print(exc)
                assert False, f'test_add_sub_scalar (sub left) failed with entry {entry}'

            # Perform the subtraction from right.
            true_result_values = other - block_storage
            result = other - sbt

            try:
                torch_equal( result.block_storage, true_result_values )
            except Exception as exc:
                print(exc)
                assert False, f'test_add_sub_scalar (sub right) failed with entry {entry}'

    def test_add_scalar_inplace(self):
        print()
        show_delimeter('test inplace-adding a scalar. ')

        # The main sbt.
        raw_sbt = SparseBlockTensor(TestSparseBlockTensor.block_shape, dtype=torch.float32)
        raw_sbt.create(shape_blocks=TestSparseBlockTensor.shape_blocks, block_indices=TestSparseBlockTensor.block_indices)
        raw_sbt.set_block_storage(TestSparseBlockTensor.values_raw, clone=False)

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
            sbt = raw_sbt.clone().to(device=device)
            scalar = entry['scalar']
            if isinstance(scalar, (int, float)):
                other = scalar
            else:
                other = scalar.to(device=device)
            block_storage = TestSparseBlockTensor.values_raw.to(device=device)

            # The result values.
            true_result_values = block_storage + other

            # Perform the addition.
            sbt.add_(other)

            try:
                torch_equal( sbt.block_storage, true_result_values )
            except Exception as exc:
                print(exc)
                assert False, f'test_add_scalar_inplace failed with entry {entry}'

    def test_sbt_2_bsr_cpu(self):
        print()
        show_delimeter('Test sbt to bsr conversion. ')

        raw_true_bsr = bsr_matrix(
            ( TestSparseBlockTensor.bsr_data, 
              TestSparseBlockTensor.bsr_indices,
              TestSparseBlockTensor.bsr_indptr ),
            shape=TestSparseBlockTensor.shape ).toarray()

        test_entries = [
            { 'device': 'cpu'  },
            { 'device': 'cuda' }
        ]

        for entry in test_entries:
            print(entry)

            device = entry['device']
            values = TestSparseBlockTensor.values_raw.to(device)

            sbt = SparseBlockTensor(TestSparseBlockTensor.block_shape, dtype=torch.float32, device=device)
            sbt.create(shape_blocks=TestSparseBlockTensor.shape_blocks, block_indices=TestSparseBlockTensor.block_indices)
            sbt.set_block_storage(values, clone=False)

            bsr = sbt_to_bsr_cpu(sbt)
            bsr = bsr.toarray()

            assert np.allclose( bsr, raw_true_bsr ), f'test_sbt_2_bsr_cpu failed with entry {entry}'

    def test_bsr_cpu_2_sbt(self):
        print()
        show_delimeter('Test bsr to sbt conversion. ')

        # Create the BSR matrix.
        bsr = bsr_matrix( 
            ( TestSparseBlockTensor.bsr_data, 
              TestSparseBlockTensor.bsr_indices, 
              TestSparseBlockTensor.bsr_indptr ), 
            shape=TestSparseBlockTensor.shape )
        
        # Convert BSR to SBM.
        sbt = bsr_cpu_to_sbt(bsr)

        # Check equality.
        dense_sbt = sbt_to_torch_sparse_coo(sbt).to_dense().permute((0,2,1,3)).numpy().reshape(
            ( TestSparseBlockTensor.shape_blocks[0]*TestSparseBlockTensor.block_shape[0], 
              TestSparseBlockTensor.shape_blocks[1]*TestSparseBlockTensor.block_shape[1] ) )
        dense_bsr = bsr.toarray()

        print(f'dense_sbt = \n{dense_sbt}')
        print(f'dense_bsr = \n{dense_bsr}')

        assert np.allclose( dense_sbt, dense_bsr ), f'test_bsr_cpu_2_sbt failed'

    def test_matmul(self):
        print()
        show_delimeter('Test matmul. ')

        # The Sparse Block Matrix.
        raw_sbt = SparseBlockTensor(TestSparseBlockTensor.block_shape, dtype=torch.float32)
        raw_sbt.create(shape_blocks=TestSparseBlockTensor.shape_blocks, block_indices=TestSparseBlockTensor.block_indices)
        raw_sbt.set_block_storage(TestSparseBlockTensor.values_raw, clone=False)

        raw_sbt = raw_sbt.coalesce()

        # The equavelent dense NumPy array.
        dense_array = sbt_to_bsr_cpu(raw_sbt).toarray()

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
            sbt = raw_sbt.to(device=device)

            # a.T @ a.
            res_at_a = sbt.transpose().coalesce() @ sbt
            res_at_a_cpu = sbt_to_bsr_cpu(res_at_a).toarray()
            assert np.allclose( res_at_a_cpu, true_res_at_a )

            # a @ a.T.
            res_a_at = sbt @ sbt.transpose().coalesce()
            res_a_at_cpu = sbt_to_bsr_cpu(res_a_at).toarray()
            assert np.allclose( res_a_at_cpu, true_res_a_at )

    def test_multiply_scalar(self):
        print()
        show_delimeter('test multiplying a scalar. ')

        # The main sbt.
        raw_sbt = SparseBlockTensor(TestSparseBlockTensor.block_shape, dtype=torch.float32)
        raw_sbt.create(shape_blocks=TestSparseBlockTensor.shape_blocks, block_indices=TestSparseBlockTensor.block_indices)
        raw_sbt.set_block_storage(TestSparseBlockTensor.values_raw, clone=False)

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
            sbt = raw_sbt.to(device=device)
            scalar = entry['scalar']
            if isinstance(scalar, (int, float)):
                other = scalar
            else:
                other = scalar.to(device=device)
            block_storage = TestSparseBlockTensor.values_raw.to(device=device)

            # The result values.
            true_result_values = block_storage * other

            # Perform the multiplication from left.
            result = sbt * other

            try:
                torch_equal( result.block_storage, true_result_values )
            except Exception as exc:
                print(exc)
                assert False, f'test_multiply_scalar (left) failed with entry {entry}'

            # Perform the multiplication from right.
            result = other * sbt

            try:
                torch_equal( result.block_storage, true_result_values )
            except Exception as exc:
                print(exc)
                assert False, f'test_multiply_scalar (right) failed with entry {entry}'

    def test_multiply_scalar_inplace(self):
        print()
        show_delimeter('test inplace multiplying a scalar. ')

        # The main sbt.
        raw_sbt = SparseBlockTensor(TestSparseBlockTensor.block_shape, dtype=torch.float32)
        raw_sbt.create(shape_blocks=TestSparseBlockTensor.shape_blocks, block_indices=TestSparseBlockTensor.block_indices)
        raw_sbt.set_block_storage(TestSparseBlockTensor.values_raw, clone=False)

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
            sbt = raw_sbt.clone().to(device=device)
            scalar = entry['scalar']
            if isinstance(scalar, (int, float)):
                other = scalar
            else:
                other = scalar.to(device=device)
            block_storage = TestSparseBlockTensor.values_raw.to(device=device)

            # The result values.
            true_result_values = block_storage * other

            # Perform the multiplication from left.
            sbt.mul_(other)

            try:
                torch_equal( sbt.block_storage, true_result_values )
            except Exception as exc:
                print(exc)
                assert False, f'test_inplace_multiply_scalar failed with entry {entry}'

if __name__ == '__main__':
    import os
    print('Run %s. ' % (os.path.basename(__file__)))
    unittest.main()