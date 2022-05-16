
class OptimizationAlgorithm(object):
    def __init__(self):
        super().__init__()

        # The PyTorch Module representing the optimization layer.
        self.optim_layer = None

        # Solver configuration/properties.
        self.properties = None # A class or a dict?

        # Logging (for debugging?)
        self.logger = None

    # Interfacing functions.

    def initialize(self):
        '''
        The equivalent of g2o OptimizationAlgorithm::init().
        '''
        raise NotImplementedError()

    def solve(self, iteration: int):
        raise NotImplementedError()

    # Normal member functions.
    # None at the moment.
