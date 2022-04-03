
# Unit test tools.
import functools
import inspect
import unittest

# System tools.
import numpy as np

# PyTorch
import torch

# The test subject.
from pypose.module import ba_layer

# Construct a convenient helper function.
torch_equal = functools.partial( 
    torch.testing.assert_close, atol=0, rtol=1e-6 )

def show_delimeter(msg):
    print(f'========== {msg} ==========')

class Test_ba_layer_standalone_functions(unittest.TestCase):
    def test_sparse_eye(self):
        print()
        show_delimeter( inspect.stack()[0][3] )

        args = [
            [ torch.float, 'cpu', 'float' ],
            [ torch.int, 'cpu', 'int' ],
            [ torch.float, 'cuda', 'float' ],
            [ torch.int, 'cuda', 'int' ],
        ]

        for arg in args:
            print(arg)

            # A float dense matrix.
            dense = torch.eye(5, dtype=arg[0], device=arg[1])

            # Create a 5x5 sparse diagnal matrix.
            smat = ba_layer.sparse_eye(5, dtype=arg[0], device=arg[1])

            try:
                torch_equal( smat.to_dense(), dense )
            except Exception as exc:
                print(exc)
                self.assertTrue(False, f'sparse_eye failed with {arg[2]} type on {arg[1]}')

    def test_nearest_log2(self):
        print()
        show_delimeter( inspect.stack()[0][3] )

        # Input-output order.
        test_entries = [
            [ 2, 1 ],
            [ 3, 1 ],
            [ 4, 2 ],
            [ 1024, 10 ],
            [ 2047, 10 ],
        ]

        for entry in test_entries:
            print(entry)

            self.assertEqual(
                ba_layer.nearest_log2(entry[0]),
                entry[1],
                f'nearest log2 of {entry[0]} should be {entry[1]}'
            )

    def test_separate_power2(self):
        print()
        show_delimeter( inspect.stack()[0][3] )

        # Input-output order.
        test_entries = [
            [ 2, ( 2, 0 ) ],
            [ 3, ( 2, 1 ) ],
            [ 4, ( 2, 2 ) ],
            [ 7, ( 4, 3 ) ],
            [ 8, ( 4, 4 ) ],
            [ 9, ( 4, 5 ) ],
        ]

        for entry in test_entries:
            print(entry)

            self.assertEqual(
                ba_layer.separate_power2( entry[0] ),
                entry[1],
                f'separated value of {entry[0]} shoudl be {entry[1]}'
            )

    def test_sparse_matrix_power(self):
        print()
        show_delimeter( inspect.stack()[0][3] )

        # A = [
        #   1, 0, 1
        #   0, 1, 0
        #   1, 0, 1 ]

        indices = [
            [ 0, 0, 1, 2, 2],
            [ 0, 2, 1, 0, 2]
        ]

        values = [ 1, 1, 1, 1, 1 ]

        # Test entries.
        test_entries = [
            { 'dtype': torch.float, 'device': 'cpu', 'ep': 0 },
            { 'dtype': torch.float, 'device': 'cpu', 'ep': 1 },
            { 'dtype': torch.float, 'device': 'cpu', 'ep': 2 },
            { 'dtype': torch.float, 'device': 'cpu', 'ep': 3 },
            { 'dtype': torch.float, 'device': 'cpu', 'ep': 4 },
            { 'dtype': torch.float, 'device': 'cpu', 'ep': 5 },
            { 'dtype': torch.float, 'device': 'cuda', 'ep': 0 },
            { 'dtype': torch.float, 'device': 'cuda', 'ep': 1 },
            { 'dtype': torch.float, 'device': 'cuda', 'ep': 2 },
            { 'dtype': torch.float, 'device': 'cuda', 'ep': 3 },
            { 'dtype': torch.float, 'device': 'cuda', 'ep': 4 },
            { 'dtype': torch.float, 'device': 'cuda', 'ep': 5 },
        ]

        for entry in test_entries:
            print(entry)

            # Compose the sparse matrix.
            A = torch.sparse_coo_tensor( indices, values, (3, 3), dtype=entry['dtype'], device=entry['device'] )

            # Get the dense matrix.
            D = A.to_dense()

            # Compute the power.
            PA = ba_layer.sparse_matrix_power(A, entry['ep'])
            PD = torch.linalg.matrix_power(D, entry['ep'])

            # Compare
            try:
                torch_equal( PA.to_dense(), PD )
            except Exception as exc:
                print(exc)
                self.assertTrue(False, f'sparse_matrix_power failed with entry {entry}')

    def test_find_leading_feature_from_full_matches(self):
        print()
        show_delimeter( inspect.stack()[0][3] )

        # The full matches among features.
        raw_full_matches = [
            [ 0, 0, 1, 2 ],
            [ 1, 4, 4, 3 ],
        ]

        # Test entries.
        test_entries = [
            { 'index': 0, 'device': 'cpu', 'expected': 0 },
            { 'index': 1, 'device': 'cpu', 'expected': 0 },
            { 'index': 2, 'device': 'cpu', 'expected': 2 },
            { 'index': 3, 'device': 'cpu', 'expected': 2 },
            { 'index': 4, 'device': 'cpu', 'expected': 0 },
            { 'index': 0, 'device': 'cuda', 'expected': 0 },
            { 'index': 1, 'device': 'cuda', 'expected': 0 },
            { 'index': 2, 'device': 'cuda', 'expected': 2 },
            { 'index': 3, 'device': 'cuda', 'expected': 2 },
            { 'index': 4, 'device': 'cuda', 'expected': 0 },
        ]

        for entry in test_entries:
            print(entry)

            # Create a Tensor.
            full_matches = torch.Tensor( raw_full_matches ).to(device=entry['device'])

            leading_index = ba_layer.find_leading_feature_from_full_matches( entry['index'], full_matches )

            expected_tensor = torch.Tensor( [entry['expected']] ).type(torch.int64).to(device=entry['device'])

            self.assertEqual( 
                leading_index, 
                expected_tensor, 
                f'find_leading_feature_from_full_matche failed with entry {entry}' )

class Test_ba_layer(unittest.TestCase):
    common_ba_layer_obj = None

    @classmethod
    def setUpClass(cls):
        print('Test_ba_layer set up. ')
        cls.common_ba_layer_obj = ba_layer.BALayer( 2, 1 )

    def test_associate_3d_point_indices_2_features(self):
        print()
        show_delimeter( inspect.stack()[0][3] )

        # The full matches among features.
        raw_full_matches = [
            [ 0, 0, 1, 2 ],
            [ 1, 4, 4, 3 ],
        ]

        # The correct association.
        raw_association = [
            0, 0, 1, 1, 0
        ]

        # Test entries.
        test_entries = [
            { 'device': 'cpu' },
            { 'device': 'cuda' },
        ]

        for entry in test_entries:
            print(entry)

            # Create full_matches Tensor.
            full_matches = torch.Tensor( raw_full_matches ).type(torch.int64).to(device=entry['device'])

            # Create expected result.
            expected_tensor = torch.Tensor( raw_association ).type(torch.int64).to(device=entry['device'])

            # Find the association.
            association = Test_ba_layer.common_ba_layer_obj.associate_3d_point_indices_2_features(
                5, full_matches )

            # Compare
            try:
                torch_equal( association, expected_tensor )
            except Exception as exc:
                print(exc)
                self.assertTrue(False, f'BALayer.associate_3d_point_indices_2_features failed with entry {entry}')

    def test_find_3d_point_association(self):
        print()
        show_delimeter( inspect.stack()[0][3] )

        # Not fully-matched.
        raw_matches = [
            [ 0, 1 ],
            [ 1, 4 ],
            [ 4, 5 ],
            [ 2, 3 ],
        ]

        # True association.
        raw_true_association = [ 0, 0, 1, 1, 0, 0 ]

        # Other information.
        n_img  = 4
        n_feat = 6

        # Test entries.
        test_entries = [
            { 'device': 'cpu' },
            { 'device': 'cuda' },
        ]

        for entry in test_entries:
            print(entry)

            # Create the tracks Tensor.
            matches = torch.Tensor( raw_matches ).type(torch.int64).to(device=entry['device'])

            # True association.
            expected = torch.Tensor( raw_true_association ).type(torch.int64).to(device=entry['device'])

            # Find the association.
            association = Test_ba_layer.common_ba_layer_obj.find_3d_point_association(n_img, n_feat, matches)

            # Compare
            try:
                torch_equal( association, expected )
            except Exception as exc:
                print(exc)
                self.assertTrue(False, f'BALayer.find_3d_point_association failed with entry {entry}')

