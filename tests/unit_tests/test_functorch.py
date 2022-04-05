# Unit test tools.
import functools
import inspect
import unittest

# System tools.
import numpy as np

# Test utils.
from tests.unit_tests.common import ( torch_equal, show_delimeter )

# PyTorch
import torch
from functorch import vmap, jacrev

def huber_cost(e, delta=0.1):
    n = torch.norm(e)

    if n <= delta:
        return 0.5 * n

    return delta * ( torch.sqrt(n) - 0.5 * delta )

def huber_cost_v(e, delta=torch.Tensor([0.1])):
    n = torch.norm(e)
    m = n <= delta
    m = m.type(torch.float)
    return m * ( 0.5 * n ) + (1 - m) * delta * ( torch.sqrt(n) - 0.5 * delta )

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