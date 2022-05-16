
from .optimization_algorithm_with_hessian import OptimizationAlgorithmWithHessian

class OptimizationAlgorithmLevenberg(OptimizationAlgorithmWithHessian):
    def __init__(self):
        super.__init__()

        self.currnt_lambda = None # Float, g2o _currentLambda.
        self.user_lambda_init = None # Float, g20 _userLambdaInit.
        self.max_trials_after_failure = 10 # int.
        self.levenberg_iterations = 0 # int.
        self.tau = 1e-5
        self.goog_step_lower_scale = 1. / 3
        self.goog_step_upper_scale = 2. / 3

    def solve(self):

        # Build stucture for the block solver if it is the first iteration.

        # Compute the errors.

        # Compute Chi2.

        # Build the system for the block solver.

        # Compute the initial lambda value if it is the first iteration.

        # Begin the LM iteration until termination conditions/events are reached.

        # Push current state?

        # Set lambda for the block solver.

        # Solve the sparse block system.

        # Update the solution.

        # Recomputed the diagnal?

        # Compute the error.

        # Compute the rho value.

        # Update the lambda value based on rho.

        # Discard or keep the current solution based on rho.

        raise NotImplementedError()

    def _compute_lambda_init(self):
        
        # Compute diagnal of the Hessians and figure out lambda value with tau.

        raise NotImplementedError()

    def _compute_scale(self):

        # Compute the scale with the current solution and RHS.

        raise NotImplementedError()

    
