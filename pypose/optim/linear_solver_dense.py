
# PyTorch
import torch

# Local package.
from .linear_solver import LinearSolver

class LinearSolverDense(LinearSolver):
    def __init__(self) -> None:
        super().__init__()

        self.reset = True
        self.H = None

    def initialize(self):
        # Do nothing.
        pass

    def solve(self, A, b):
        '''
        Solve Ax=b by treating A as a dense matrix.

        A (Tensor): The sparse/dense coefficient matrix.
        b (Tensor): The right-hand-side vector.

        Returns:
        x
        '''

        # Prepare or reallocated self.H.

        # Conver A to self.H.

        # Solve for x.
        # torch.linalg.cholesky()
        # torch.cholesky_solve()

