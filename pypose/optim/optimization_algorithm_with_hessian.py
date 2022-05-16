
from .optimization_algorithm import OptimizationAlgorithm

class OptimizationAlgorithmWithHessian(OptimizationAlgorithm):
    def __init__(self):
        super().__init__()

        self.solver = None # The block solver.

    def initialize(self):
        '''
        Figure out if we need to use Schur complement.
        '''
        raise NotImplementedError()

    def build_linear_structure(self):
        '''
        Build the structure for the block solver.
        '''
        raise NotImplementedError()

    def update_linear_system():
        '''
        Build the system for the block solver.
        '''
        raise NotImplementedError()

