import torch, warnings
from torch import Tensor, nn
from torch.linalg import pinv, lstsq, cholesky_ex

# CuPy.
import cupy as cp
import cupyx as cpx
from cupyx.scipy.sparse.linalg import spsolve

# Local package.
from pypose.sparse.sparse_block_tensor import (
    SparseBlockTensor, 
    sbt_to_cupy, torch_to_cupy, cupy_to_torch )


# NOTE: The base class is not nn.Module.
class LinearSolverCuSparse(object):
    def __init__(self) -> None:
        super().__init__()

    def initialize(self):
        # Do nothing.
        pass

    def solve(self, A, b):
        '''
        Solve Ax=b by converting A to a CuPy sparse matrix.

        A (SparseBlockMatrix): The coefficient matrix.
        b (Tensor): The right-hand-side vector.

        Returns:
        x (Tesnor)
        '''

        assert A.is_cuda, f'Only supports SparseBlockMatrices on GPU. '
        assert b.is_cuda, f'Only supports Tensors on GPU. '

        # Convert things to CuPy.
        cu_A = sbt_to_cupy(A)
        cu_b = torch_to_cupy(b)

        # Solve.
        cu_x = spsolve(cu_A, cu_b)

        # Convert x back to PyTorch.
        return cupy_to_torch(cu_x)