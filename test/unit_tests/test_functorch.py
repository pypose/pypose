# Unit test tools.
import functools
import inspect
import unittest

# System tools.
import numpy as np

# Test utils.
from .common import ( torch_equal, show_delimeter )

# PyTorch
import torch
from functorch import vmap, jacrev

def huber_cost(e, delta=0.1):
    '''
    This is the definition used by Ceres Solver.
    http://ceres-solver.org/nnls_modeling.html#instances
    '''
    n = torch.norm(e)

    if n <= delta:
        return 0.5 * n

    return delta * ( torch.sqrt(n) - 0.5 * delta )

def huber_cost_v(e, delta=torch.Tensor([0.1])):
    '''
    This is the definition used by Ceres Solver.
    http://ceres-solver.org/nnls_modeling.html#instances
    '''
    n = torch.norm(e)
    m = n <= delta
    m = m.type(torch.float)
    return m * ( 0.5 * n ) + (1 - m) * delta * ( torch.sqrt(n) - 0.5 * delta )

class MyTensor(torch.Tensor):
    def __init__(self, *args, **kwargs):
        pass

    @staticmethod
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls, *args, **kwargs)

    def my_func(self):
        return self
class Test_functorch(unittest.TestCase):
    def test_robust_cost_function(self):
        print()
        show_delimeter('Test robust cost function. ')

        e = torch.rand((10, 2))

        jacobian = vmap(jacrev(huber_cost_v))(e)

        # print(jacobian)
        # print(jacobian.shape)

        e = e.detach().clone()
        e.requires_grad = True

        costs = []
        for ee in e:
            costs.append ( huber_cost(ee) )

        cost_tensor = torch.stack( costs, dim=0 )
        cost_tensor.backward( torch.ones((10,)) )
        # print(e.grad.unsqueeze(1))

        # Compare
        try:
            torch_equal( jacobian, e.grad.unsqueeze(1) )
        except Exception as exc:
            print(exc)
            self.assertTrue(False, 
                f'robust cost function failed with \
                e = {e}\njacobian = {jacobian} \
                e.grad.unsqueeze(1) = {e.grad.unsqueeze(1)} ')

    def test_inherited_type(self):
        print()
        show_delimeter('Test functorch with inherited type. ')

        # x = MyTensor((0.5,))
        # y = torch.Tensor((1,))

        # f = lambda a, b: a.my_func() * b
        # j = jacrev(f)(x, y)

    def test_non_tensor_type(self):
        print()
        show_delimeter('Test non tensor type')

        # Two sparse block matrices.
        block_indices_0 = [
            [0, 0, 0],
            [0, 1, 1],
            [0, 2, 2],
            [1, 1, 3],
            [2, 0, 4],
            [2, 2, 5],
        ]

        block_indices_1 = [
            [0, 0, 0],
            [0, 2, 1],
            [1, 0, 2],
            [1, 1, 3],
            [2, 0, 4],
            [2, 2, 5],
        ]

        # Compose the block_cols representations.
        block_cols_0 = [ dict() for _ in range(3) ]
        for block in block_indices_0:
            block_cols_0[ block[1] ][ block[0] ] = block[2]

        block_cols_1 = [ dict() for _ in range(3) ]
        for block in block_indices_1:
            block_cols_1[ block[1] ][ block[0] ] = block[2]

        print(f'block_cols_0 = \n{block_cols_0}')
        print(f'block_cols_1 = \n{block_cols_1}')

        def print_list(v):
            print(v)

        vmap(print_list)(block_cols_0)

if __name__ == '__main__':
    import os
    print('Run %s. ' % (os.path.basename(__file__)))
    unittest.main()