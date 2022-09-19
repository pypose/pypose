import pypose as pp
import torch as torch


class LTINoise(pp.module.LTI):
    r"""
    Discrete-time Linear Time-Invariant (LTI) system with process and measurement noise. 
    See :obj:`LTI` for more information.

    Args:
        Q (:obj:`Tensor`): The process noise covariance
        R (:obj:`Tensor`): The measurement noise covariance
    """

    def __init__(self, A, B, C, D, Q, R, c1=None, c2=None):
        super(LTINoise, self).__init__(A, B, C, D, c1, c2)
        # both are square matricies
        self.register_buffer("Q", torch.eye(A.shape) if Q is None else Q)
        self.register_buffer("R", torch.eye(C.shape[0]) if R is None else R)

    def state_transition(self, state, input):
        r"""
        Perform one step of LTI state transition.

        .. math::
            \mathbf{w} \sim (0, \mathbf{Q}) \\
            \mathbf{z} = \mathbf{A}\mathbf{x} + \mathbf{B}\mathbf{u} + \mathbf{c}_1 + \mathbf{w}
        
        """
        w = self.Q @ torch.randn(state.shape)  # gaussian noise
        return super().state_transition(state, input) + w

    def observation(self, state, input):
        r"""
        Return the observation of LTI system at current time step.

        .. math::
            \mathbf{v} \sim (0, \mathbf{R}) \\
            \mathbf{y} = \mathbf{C}\mathbf{x} + \mathbf{D}\mathbf{u} + \mathbf{c}_2 + \mathbf{v}
        
        """
        v = self.R @ torch.randn(self.R.shape)  # gaussian noise
        return super().observation(state, input) + v
