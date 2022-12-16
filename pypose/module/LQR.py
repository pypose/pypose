import torch as torch
import time
import torch.nn as nn
import pypose as pp


class DPLQR(nn.Module):
    r'''
    Discrete-time finite-horizon Linear Quadratic Regulator (LQR)

    Args:
        n_state (int): Parameter that determines the dimension of the state.
        n_ctrl (int): Parameter that determines the dimension of the input.
        T (int): Total time steps.
        system (instance): The system sent to LQR to affine state-transition dynamics.

    LQR finds the optimal nominal trajectory :math:`\mathbf{\tau}_{1:T}^*` = 
    :math:`\begin{Bmatrix} \mathbf{x}_t, \mathbf{u}_t \end{Bmatrix}_{1:T}` 
    of the optimization problem

    .. math::
        \begin{aligned}
            \mathbf{\tau}_{1:T}^* = \mathop{\arg\min}\limits_{\tau_{1:T}} \sum\limits_t\frac{1}{2}
            \mathbf{\tau}_t^\top\mathbf{Q}_t\mathbf{\tau}_t + \mathbf{p}_t^\top\mathbf{\tau}_t \\
            \mathrm{s.t.} \quad \mathbf{x}_1 = \mathbf{x}_{init}, 
            \ \mathbf{x}_{t+1} = \mathbf{F}_t\mathbf{\tau}_t + \mathbf{f}_t \\
        \end{aligned}

    where :math:`\mathbf{\tau}` = :math:`\begin{bmatrix} \mathbf{x} & \mathbf{u} \end{bmatrix}^\top`, 
    :math:`\mathbf{F}` = :math:`\begin{bmatrix} \mathbf{A} & \mathbf{B} \end{bmatrix}`, 
    :math:`\mathbf{f}` = :math:`\mathbf{c}_1`.

    From a policy learning perspective, this can be interpreted as a module with unknown parameters
    :math:`\begin{Bmatrix} \mathbf{Q}, \mathbf{p}, \mathbf{F}, \mathbf{f} \end{Bmatrix}`, 
    which can be integrated into a larger end-to-end learning system.

    Note:
        Here, we consider the system sent to LQR as a linear system in each small horizon. 
        There are many ways of solving the LQR problem, such as solving Riccati equation, 
        dynamic programming (DP), etc. Here, we implement the DP method.
        
        For more details about mathematical process, please refer to 
        http://rll.berkeley.edu/deeprlcourse/f17docs/lecture_8_model_based_planning.pdf

    Example:
        >>> n_batch = 2
        >>> n_state, n_ctrl = 4, 3
        >>> n_sc = n_state + n_ctrl
        >>> T = 5
        >>> Q = torch.randn(n_batch, T, n_sc, n_sc)
        >>> Q = torch.matmul(Q.mT, Q)
        >>> p = torch.randn(n_batch, T, n_sc)
        >>> A = torch.tile(torch.eye(n_state) + 0.2 * torch.randn(n_state, n_state), (n_batch, 1, 1))
        >>> B = torch.randn(n_batch, n_state, n_ctrl)
        >>> C = torch.tile(torch.eye(n_state), (n_batch, 1, 1))
        >>> D = torch.tile(torch.zeros(n_state, n_ctrl), (n_batch, 1, 1))
        >>> c1 = torch.tile(torch.randn(n_state), (n_batch, 1))
        >>> c2 = torch.tile(torch.zeros(n_state), (n_batch, 1))
        >>> x_init = torch.randn(n_batch, n_state)
        >>> lti = pp.module.LTI(A, B, C, D, c1, c2)
        >>> LQRDP = pp.module.DPLQR(lti)
        >>> x_lqr, u_lqr, objs_lqr, tau = LQRDP.forward(x_init, Q, p)
        >>> print(x_lqr)
        >>> print(u_lqr)
        tensor([[[-0.5453, -0.5922, -1.2467,  0.8291],
                 [ 0.2576, -0.5661, -0.9135,  1.1018]],
                [[-2.9563,  3.2528,  0.6299, -0.5684],
                 [[ 0.9996,  1.7761, -1.6918,  2.2817]],
                [[ 1.7806,  7.9820,  4.4041, -4.8167],
                 [[ 2.2647,  5.1426, -0.9377,  0.7258]],
                [[-7.0567,  2.4997, -3.3157,  3.5961],
                 [[-4.1525,  5.0354, -2.7758,  4.3140]],
                [[-5.1267,  6.8131, -0.6195,  3.7960],
                 [[-0.2206,  9.1061, -0.0106,  3.7945]]]
        tensor([[[-4.1313,  6.6788, -2.7602],
                 [ 0.0752,  1.1968, -0.2460]],
                [[ 2.6432,  1.6548, -3.2089],
                 [-1.7543,  4.5867, -4.6340]],
                [[-3.4785, -3.5352,  1.6931],
                 [-6.5703,  5.4751, -3.8288]],
                [[-0.2215,  4.0535, -3.0925],
                 [ 0.5750,  3.5659, -5.1008]],
                [[ 4.9618,  6.6287, -7.5575],
                 [ 0.3090, -3.1843,  0.7678]]] 
    '''


    def __init__(self, system):
        super(DPLQR, self).__init__()

        self.system = system
        self.c1 = system.c1
        self.time = system.systime
            

    def forward(self, x_init, Q, p):
        r'''
        Perform one step advance for the LQR problem.
        '''
        Ks, ks = self.lqr_backward(Q, p)
        new_x, new_u, objs, tau = self.lqr_forward(x_init, Q, p, Ks, ks)
        #new_x, new_u = self.lqr_forward(x_init, Q, p, Ks, ks)
        return new_x, new_u, objs, tau

    
    def lqr_backward(self, Q, p):
        r'''
        The backward recursion of the dynamic programming to solve LQR.

        Args:
            Q (:obj:`Tensor`): The matrix of quadratic parameter.
            p (:obj:`Tensor`): The constant vector of quadratic parameter.

        Returns:
            Tuple of Tensor: The status feedback controller :math:`\mathbf{K}_s` and 
            :math:`\mathbf{k}_s` at all steps.
        '''
        T = Q.size(-3)
        Ks = []
        ks = []
        Vtp1 = vtp1 = None
        n_state = self.system.B.size(-2)
        n_ctrl = self.system.B.size(-1)
        
        for t in range(T-1, -1, -1): 
            pt = p[:,t,:]
            pt = pt.unsqueeze(1)

            if self.c1.ndim ==3:
                c1 = self.c1[:,t,:]
                c1 = c1.unsqueeze(1)
            elif self.c1.ndim == 2:
                c1 = self.c1.unsqueeze(1)
            elif self.c1.ndim == 1:
                c1 = self.c1.unsqueeze(0)

            if self.system.A.ndim == 4:
                A = self.system.A[:,t,:,:]
                B = self.system.B[:,t,:,:]
                F = torch.cat((A, B), axis = 2)
            elif self.system.A.ndim == 3:
                F = torch.cat((self.system.A, self.system.B), axis = 2)
            elif self.system.A.ndim == 2:
                F = torch.cat((self.system.A, self.system.B), dim = 1)

            if t == T-1:
                Qt = Q[:,t,:,:]
                qt = pt.mT
            else:
                Ft = F
                Ft_T = Ft.mT
                Qt = Q[:,t,:,:] + Ft_T @ Vtp1 @ Ft
                if self.c1 is None or self.c1.numel() == 0:
                    qt = pt.mT + Ft_T @ vtp1
                else:
                    ft = c1.mT
                    qt = pt.mT + Ft_T @ Vtp1 @ ft + Ft_T @ vtp1
            
            Qt_xx = Qt[:, :n_state, :n_state]
            Qt_xu = Qt[:, :n_state, n_state:]
            Qt_ux = Qt[:, n_state:, :n_state]
            Qt_uu = Qt[:, n_state:, n_state:]
            qt_x = qt[:, :n_state, :]
            qt_u = qt[:, n_state:, :]

            if n_ctrl == 1:
                Kt = -(1./Qt_uu)*Qt_ux
                kt = -(1./Qt_uu)*qt_u
            else:
                Qt_uu_inv = [torch.linalg.pinv(Qt_uu[i]) for i in range(Qt_uu.shape[0])]
                Qt_uu_inv = torch.stack(Qt_uu_inv)
                Kt = -Qt_uu_inv.bmm(Qt_ux)
                kt = -Qt_uu_inv @ qt_u

            Kt_T = Kt.mT

            Ks.append(Kt)
            ks.append(kt)

            Vtp1 = Qt_xx + Qt_xu @ Kt + Kt_T @ Qt_ux + Kt_T @ Qt_uu @ Kt
            vtp1 = qt_x + Qt_xu @ kt + Kt_T @ qt_u + Kt_T @ Qt_uu @ kt

        return Ks, ks

    def lqr_forward(self, x_init, Q, p, Ks, ks):
        r'''
        The forward recursion of the dynamic programming to solve LQR.

        .. math::
            \mathbf{u}_t = \mathbf{K}_t\mathbf{x}_t + \mathbf{k}_t

        Args:
            Q (:obj:`Tensor`): The matrix of quadratic parameter.
            p (:obj:`Tensor`): The constant of quadratic parameter.
            Ks (:obj:`Tensor`): The matrix of status feedback controller at all steps.
            ks (:obj:`Tensor`): The constant of status feedback controller at all steps.

        Returns:
            Tuple of Tensor: The state of the dynamical system :math:`\mathbf{x}`, 
            the input to the dynamical system :math:`\mathbf{u}`,
            the costs to the dynamical system :math:`\mathbf{c}`, 
            and the optimal nominal trajectory :math:`\mathbf{\tau}` to the dynamical system.
        '''
        T = Q.size(-3)
        new_u = []
        new_x = [x_init]
        objs = []
        tau = []
        ks = ks
        Ks = Ks


        for t in range(T):
            t_rev = T-1-t
            Kt = Ks[t_rev]
            kt = ks[t_rev]
            new_xt = new_x[t]
            new_ut = torch.einsum('...ik,...k->...i', [Kt, new_xt]) + kt.mT.squeeze()
            new_u.append(new_ut)

            new_xut = torch.cat((new_xt, new_ut), dim=1)
            if t < T-1:
                new_xtp1, y_system = self.system(new_xt, new_ut)
                new_x.append(new_xtp1)

            """ obj = 0.5*new_xut.unsqueeze(1).matmul(Q[:,t,:,:]).matmul(new_xut.unsqueeze(2)).squeeze(1).squeeze(1) + \
                    torch.matmul(new_xut.unsqueeze(1), p[:,t,:].unsqueeze(2)).squeeze(1).squeeze(1) """
            obj = 0.5 * new_xut @ Q[:,t,:,:] @ new_xut.mT + new_xut @ p[:,t,:].mT
            objs.append(obj)

            tau.append(new_xut)
            

        new_u = torch.stack(new_u)
        new_x = torch.stack(new_x)
        objs = torch.stack(objs) 

        return new_x, new_u, objs, tau
