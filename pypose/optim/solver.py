
class Solver(object):
    def __init__(self):
        super().__init__()

        self.x = None # The solution.
        self.b = None # The RHS.

        self.is_levenberg = False

    def initialize(self):
        raise NotImplementedError()

    def build_structure(self):
        raise NotImplementedError()

    def build_system(self):
        raise NotImplementedError()

    def solve(self):
        raise NotImplementedError()

    def compute_marginals(self, spinv, block_indices):
        raise NotImplementedError()

    def restore_diagnal(self):
        raise NotImplementedError()

    def vector_size(self):
        '''
        Return the size of the solution/RHS. 
        '''
        # Might be a property function.
        raise NotImplementedError()

    def levenberg(self):
        # Property function?
        return self.is_levenberg

    def support_schur(self):
        '''
        Return True if the solver supports Schur complement.
        '''
        raise NotImplementedError()

    def schur(self):
        '''
        Return True if the solver should do Schur complement.
        '''
        raise NotImplementedError()

    
