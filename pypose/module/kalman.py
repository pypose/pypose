import torch
from torch import nn
import pypose as pp


class EKF(nn.Module):
    r"""
    Linearized Extended Kalman Filter. 
    
    Args:
        model (:obj:`System`): The noisy system model to estimate using this filter. 
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, y, P, U):
        r"""
        Calculates filter one timestep.

        Args:
            x (:obj:`Tensor`): estimated system state of previous timestep
            y (:obj:`Tensor`): system observation at current timestep (measurement)
            P (:obj:`Tensor`): state covariance of previous timestep
            U (:obj:`Tensor`): system input at current timestep
        Returns:
            x (:obj:`Tensor`): estimated system state at current timestep (posteriori state estimate)
            P (:obj:`Tensor`): estimated state covariance at current timestep (posteriori covariance)

        Uses a Linearized System Model, thus it is a Linearized Extended Kalman Filter. 
        If not defined, :obj:`System.A`, :obj:`System.B`, :obj:`System.C` are computed each call using torch.autograd.functional.jacobian(), see :obj:`System`
        
        Note:
            To use this filter, your System subclass must define Q (process noise covariance) and R (measurement noise covariance). 
            Your System should add noise to the returned state and observation by the following equations:

            .. math::
                w_k \sim (0, Q) \\
                x_k = x_k + w_k

            .. math::
                v_k \sim (0, R) \\
                y_k = y_k + v_k

        
        For each :obj:`forward()` call, the Kalman Filter Equations are executed as follows:

        Note:
            Since System.x is a row-vector, before doing these calculations, x is transposed into a column vector. 

        1. Predict State (Priori State Estimate) using State Transition Equation.
        
        .. math::
            \mathbf{x} = f(x_{k-1}, U_k)
        
        For a Linear System, that is:

        .. math::
            \mathbf{x} = \mathbf{A}\mathbf{x_{k-1}} + \mathbf{B}\mathbf{U}
        
        2. Predict State Covariance.

        .. math::
            \mathbf{P} = \mathbf{A}\mathbf{P}\mathbf{A}^{T} + \mathbf{Q}
        
        3. Update Kalman Gain

        .. math::
            \mathbf{K} = \mathbf{P}\mathbf{C}^{T} (\mathbf{C}\mathbf{P}\mathbf{C}^{T} + \mathbf{R})^{-1}
        
        4. Correction Step (Posteriori State Estimate)

        .. math::
            \mathbf{x} = \mathbf{x} + \mathbf{K} (\mathbf{y} - \mathbf{C}\mathbf{x}) \\
            \mathbf{P} = (\mathbf{I} - \mathbf{K}\mathbf{C}) \mathbf{P}
        
        Example:

            1. Initialize model and filter

            >>> import torch
            >>> import pypose as pp
            >>> model = pp.module.System()  # System model
            >>> filter = pp.module.EKF(model)

            2. Calculate filter over one timestep

            >>> # u is system input
            >>> state, observation = model(state, u)  # model measurement
            >>> model.set_refpoint(state=state, input=u)
            >>> est_state, P = filter(est_state, observation, P, u)

        Note:
            Implementation is based on section 5.1 of this book

            * Dan Simon, `Optimal State Estimation: Kalman, H∞, and Nonlinear Approaches <https://onlinelibrary.wiley.com/doi/epdf/10.1002/0470045345.fmatter>`_, Cleveland State University, 2006
        

        """

        x_tposed = x.T

        # Predict
        x_tposed = self.model.state_transition(x_tposed, U)
        P = self.model.A.matmul(P).matmul(self.model.A.T) + self.model.Q
        # Update
        S = self.model.C.matmul(P).matmul(self.model.C.T) + self.model.R
        K = P.matmul(self.model.C.T).matmul(torch.linalg.inv(S))
        x_tposed = x_tposed + K.matmul(y - self.model.C.matmul(x_tposed))

        sh = K.matmul(self.model.C)  # to get the shape
        P = (torch.eye(sh.shape[0]) - K.matmul(self.model.C)).matmul(P)
        x = x_tposed.T
        return x, P


class UKF(nn.Module):
    """
    https://www.mathworks.com/matlabcentral/fileexchange/18217-learning-the-unscented-kalman-filter?s_tid=FX_rc1_behav
    https://pykalman.github.io/#unscentedkalmanfilter
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, y, P, U):
        L = x.size()
        m = y.size()
        alpha = 1e-3
        ki = 0
        beta = 2
        lmbda = alpha ** 2 * (L + ki) - L
        c = L + lmbda

    @staticmethod
    def unscented_transform(y, sgma, Wm, Wc, n, R):
        L = sgma.shape[2]
