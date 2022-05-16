
# PyTorch
import torch

# Local package.
from .linear_solver import LinearSolver

class LinearSolverPCG(LinearSolver):
    def __init__(self) -> None:
        super().__init__()

        self.tolerance = 1e-6
        self.flag_abs_tolerance = True
        self.residual = -1.0
        self.max_iter = -1

    def initialize(self):
        self.residual = -1.0
        raise NotImplementedError()

    def solve(self, A, b):
        raise NotImplementedError()

    def mult_diag(self, A, src, dst):
        raise NotImplementedError()

    def mult(self, src, dst):
        raise NotImplementedError()

    # More helper functions.

    