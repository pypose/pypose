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
import pypose as pp
from pypose.sparse.sparse_block_tensor import (
    SparseBlockTensor,
    sbt_to_cupy, torch_to_cupy, cupy_to_torch )

# from pypose.optim.solver import LinearSolverCuSparse

# # Test utils.
# from .common import ( torch_equal, torch_equal_rough, show_delimeter )

# # PyTorch
# import torch

# class Test_LinearSolverCuSparse(unittest.TestCase):

#     @classmethod
#     def setUpClass(cls) -> None:
#         #      0,  1,  2,  3,  4,  5
#         # ==========================
#         # 0 |  0,  1,  x,  x,  4,  5
#         # 1 |  2,  3,  x,  x,  6,  7
#         # 2 |  x,  x,  8,  9,  x,  x
#         # 3 |  x,  x, 10, 11,  x,  x
#         # 4 | 12, 13,  x,  x, 16, 17
#         # 5 | 14, 15,  x,  x, 18, 19

#         cls.block_shape = (2, 2)

#         cls.block_indices = [
#             [ 0, 0, 1, 2, 2 ],
#             [ 0, 2, 1, 0, 2 ]
#         ]

#         cls.rows = ( max(cls.block_indices[0]) + 1 ) * cls.block_shape[0]
#         cls.cols = ( max(cls.block_indices[1]) + 1 ) * cls.block_shape[1]
#         cls.shape = ( cls.rows, cls.cols )

#         cls.rows_block = cls.rows // cls.block_shape[0]
#         cls.cols_block = cls.cols // cls.block_shape[1]
#         cls.shape_blocks = ( cls.rows_block, cls.cols_block )

#         shape_values = ( len(cls.block_indices[0]), *cls.block_shape )
#         cls.values_raw = torch.arange(20, dtype=torch.float32).view( shape_values )

#         # Randomize the blocks.
#         for i in range( cls.values_raw.shape[0] ):
#             r = torch.rand(2).to(dtype=torch.float32).view((2,1))
#             cls.values_raw[i, 0, 0] = r[0] + 1
#             cls.values_raw[i, 0, 1] = r[1]
#             cls.values_raw[i, 1, 0] = r[1]
#             cls.values_raw[i, 1, 1] = r[0] + 1

#         cls.values_raw[3] = cls.values_raw[1]

#         cls.x = torch.rand((6, 1), dtype=torch.float32)

#     def test_solve_sym_postive_definite(self):
#         print()
#         show_delimeter('Test solving a system with symmetric postive definite coefficient matrix. ')

#         # Creawte the SparseblockMatrix.
#         device = 'cuda'
#         values = Test_LinearSolverCuSparse.values_raw.to(device)
#         sbm = SparseBlockTensor(Test_LinearSolverCuSparse.block_shape, dtype=torch.float32, device=device)
#         sbm.create(shape_blocks=Test_LinearSolverCuSparse.shape_blocks, block_indices=Test_LinearSolverCuSparse.block_indices)
#         sbm.set_block_storage(values, clone=False)

#         # Create the rhs
#         x = Test_LinearSolverCuSparse.x.to(device=device)
#         cu_x = torch_to_cupy(x)
#         cu_A = sbt_to_cupy(sbm)
#         cu_b = cu_A.dot(cu_x)

#         # Show cu_A.
#         print(f'cu_b: {cu_b.reshape((1, -1))}')
#         print(f'cu_A: \n{cu_A.toarray()}')

#         # Convert cu_b to PyTorch.
#         b = cupy_to_torch(cu_b)

#         # Solve.
#         solver = LinearSolverCuSparse()
#         solver.initialize()
#         sx = solver.solve(sbm, b).view((-1, 1))

#         # Show some information.
#         print(f'original x: {x.view((1, -1))}')
#         print(f'solved x: {sx.view((1, -1))}')

#         # Compare.
#         try:
#             torch_equal_rough( sx, x )
#         except Exception as exc:
#             print(exc)
#             self.assertTrue(False, f'test_solve_sym_postive_definite failed. ')

# if __name__ == '__main__':
#     import os
#     print('Run %s. ' % (os.path.basename(__file__)))
#     unittest.main()