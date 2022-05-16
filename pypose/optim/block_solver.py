
from .solver import Solver

class BlockSoverBase(Solver):
    def __init__(self):
        super().__init__()

    def multiply_hessian(self):
        raise NotImplementedError()

class BlockSolver(BlockSoverBase):
    def __init__(self):
        super().__init__()

        self.H_pp              = None # _Hpp.
        self.H_ll              = None # _Hll.
        self.H_pl              = None # _Hpl.
        self.H_schur           = None # _Hshcur.
        self.D_inv_schur       = None # _DInvSchur, diagonal matrix.
        self.H_pl_ccs          = None # _HplCCS, CCS matrix.
        self.H_schur_trans_ccs = None # _HschurTransposedCCS.

        # These two are only used for Schur complement.
        self.coefficients = None # _coefficients
        self.b_schur      = None # _bschur

        self.linear_solver = None # _linearSolver.

        self.do_schur = True

    def initialize(self):
        raise NotImplementedError()

    def build_structure(self):

        # Find association between camera poses and landmarks and
        # the block indices in the block sparse matrix.

        # Prepare memory for computing the Hessian.

        # Prepare for Schur complement.

        # Associate edge and memory of Hessian.

        # If do Schur complement.
        # Work on self.D_inv_schur and self.H_pl.

        # Work on self.H_schur.

        raise NotImplementedError()

    def build_system(self):

        # Work on every edge, compute the linearizedOplus.

        # Work on every vertex.

        raise NotImplementedError()

    def solve(self):

        # If Schur complement, self.linear_solver.solve(), end.

        # Build self.H_shur.

        # self.linear_solver.solve() for the camera poses.

        # Recover the land marks.

        raise NotImplementedError()

    def set_lambda(self):
        # Property function?
        raise NotImplementedError()

    def restore_diagnal(self):
        raise NotImplementedError()

    def support_schur(self):
        return True

    def schur(self):
        raise NotImplementedError()

    def multiply_hessian(self):
        raise NotImplementedError()

    def _resize(self):
        # Call self.resize_vector()

        # self.coefficients and self.b_schur.

        # self.H_pp.

        # if do Schur complement.
        # self.H_schur, self.H_ll, self.D_inv_schur, self.H_pl, self.H_pl_ccs, self.H_schur_trans_ccs.

        raise NotImplementedError()

    

    
