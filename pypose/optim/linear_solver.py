
class LinearSolver(object):
    def __init__(self) -> None:
        super().__init__()

    def initialize(self):
        '''
        If A changes, we need to call initialize().
        '''
        raise NotImplementedError()

    def solve(self, A, b):
        '''
        Solve Ax=b.
        '''

        raise NotImplementedError()

    def solve_blocks(self, A):
        raise NotImplementedError()

    def solve_pattern(self, block_indices, A):
        raise NotADirectoryError()

    
class LinearSolverCCS(LinearSolver):
    def __init__(self) -> None:
        super().__init__()

        self.block_ordering = True

    def solve_blocks(self, A):
        raise NotImplementedError()

    def solve_pattern(self, block_indices, A):
        raise NotImplementedError()
