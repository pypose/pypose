import time
import torch as torch
import torch.nn as nn
from .. import bmv, btdot

class IPDDP(nn.Module):
    r'''
    Interior-point differential dynamic programming.

    Args:
        sys (:obj:`instance`): System dynamics of the optimal control problem.
        stage_cost (:obj:`instance`): Stage cost of the optimal control problem.
        terminal_cost (:obj:`instance`): Terminal cost of the optimal control problem.
        cons (:obj:`instance`): Constraints of the optimal control problem.
        T (:obj:`int`): Horizon of the optimal control problem.
        B (:obj:`tuple`): Batch size of the optimal control problem, default is (1,).

    A discrete-time system can be described as:

    .. math::
        \begin{align*}
            \mathbf{x}_{t+1} &= \mathbf{f}(\mathbf{x}_{t}, \mathbf{u}_{t}, t)              \\
        \end{align*}

    where :math:`\mathbf{x}`, :math:`\mathbf{u}` are the state and input of the general nonlinear system.
    The subscript :math:`\cdot_{t}` denotes the time step.

    IPDDP finds the optimal nominal trajectory :math:`\mathbf{\tau}^*` =
    :math:`\begin{Bmatrix} \mathbf{x}_t, \mathbf{u}_t \end{Bmatrix}_{0:T-1} \cup \begin{Bmatrix} \mathbf{x}_T \end{Bmatrix}`
    for the following optimization problem:

    .. math::
        \begin{align*}
          \mathbf{\tau}^* = \mathop{\arg\min}\limits_{\tau} & \sum\limits_{t=0}^{T-1}q(\mathbf{\tau}_t) + p(\mathbf{\tau}_T) \\
          \mathrm{s.t.} \quad \mathbf{x}_0 &= \mathbf{x}_{\text{init}}, \\
          \mathbf{x}_{t+1} &= \mathbf{f}(\mathbf{x}_{t}, \mathbf{u}_{t}, t),  \\
          \mathbf{c}(\mathbf{\tau}_t) &\leq \mathbf{0}.
        \end{align*}

    where :math:`\mathbf{\tau}_t` = :math:`\begin{bmatrix} \mathbf{x}_t \\ \mathbf{u}_t
    \end{bmatrix}`; :math:`q` and :math:`p` denote the stage and terminal costs, respectively;
    :math:`\mathbf{c}` is the stage-wise inequality constraints. In addition to the notations
    defined above, two additional variables :math:`\mathbf{s}` and :math:`\mathbf{y}` are
    introduced, which denote the dual variable and the slack variable, respectively.
    We use :math:`\mathrm{infeas} = 0` (resp., :math:`\mathrm{infeas} = 1`) to denote
    the case where the initial trajectory is feasible (resp., infeasible).

    The IPDDP process can be summarised as iterative backward and forward recursions.

    - The backward recursion.

      For :math:`t` = :math:`T-1` to 0:

        .. math::
            \begin{align*}
                \mathbf{Q}_t &= \begin{cases}
                \mathbf{Q}_t + \mathbf{F}_t^\top\mathbf{V}_{t+1}\mathbf{F}_t
                                    + \mathbf{v}_{t+1} \odot \mathbf{G}_t
                                    - \mathbf{W}_t^\top \mathbf{S}_t \mathbf{C}_t^{-1} \mathbf{W}_t, \mathrm{if ~infeas} = 0 \\
                 \mathbf{Q}_t + \mathbf{F}_t^\top\mathbf{V}_{t+1}\mathbf{F}_t
                                    + \mathbf{v}_{t+1} \odot \mathbf{G}_t
                                    + \mathbf{W}_t^\top \mathbf{S}_t \mathbf{Y}_t^{-1} \mathbf{W}_t, \mathrm{if ~infeas} = 1 \\
                                \end{cases}                           \\
                \mathbf{q}_t &= \begin{cases}
                \mathbf{q}_t + \mathbf{F}_t^\top\mathbf{v}_{t+1}
                                          + \mathbf{W}_t^\top\mathbf{s}_t
                                          - \mathbf{W}_t^\top \mathbf{C}_t^{-1}\mathbf{r}_t, \mathrm{if ~infeas} = 0 \\
                \mathbf{q}_t + \mathbf{F}_t^\top\mathbf{v}_{t+1}
                                          + \mathbf{W}_t^\top\mathbf{s}_t
                                          + \mathbf{W}_t^\top \mathbf{Y}_t^{-1}\hat{\mathbf{r}}_t, \mathrm{if ~infeas} = 1 \\
                                \end{cases}            \\
                \mathbf{K}_t &= -\mathbf{Q}_{\mathbf{u}_t, \mathbf{u}_t}^{-1}
                                                     \mathbf{Q}_{\mathbf{u}_t, \mathbf{x}_t} \\
                \mathbf{k}_t &= -\mathbf{Q}_{\mathbf{u}_t, \mathbf{u}_t}^{-1}
                                                           \mathbf{q}_{\mathbf{u}_t}         \\
                \mathbf{K}_t^{\mathbf{s}} &= \begin{cases}
                - \mathbf{S}_t\mathbf{C}_t^{-1} (\mathbf{c}_{\mathbf{x}_t}
                                        + \mathbf{c}_{\mathbf{u}_t} \mathbf{K}_t), \mathrm{if ~infeas} = 0 \\
                \mathbf{S}_t\mathbf{Y}_t^{-1} (\mathbf{c}_{\mathbf{x}_t}
                                        + \mathbf{c}_{\mathbf{u}_t} \mathbf{K}_t), \mathrm{if ~infeas} = 1 \\
                                \end{cases}
                                 \\
                \mathbf{k}_t^{\mathbf{s}} &= \begin{cases}
                - \mathbf{C}_t^{-1} (\mathbf{r}_t + \mathbf{S}_t
                                                    \mathbf{c}_{\mathbf{u}_t} \mathbf{k}_t), \mathrm{if ~infeas} = 0 \\
                \mathbf{Y}_t^{-1} (\hat{\mathbf{r}}_t + \mathbf{S}_t
                                                    \mathbf{c}_{\mathbf{u}_t} \mathbf{k}_t), \mathrm{if ~infeas} = 1 \\
                                \end{cases}                 \\
                \mathbf{K}_t^{\mathbf{y}} &= \begin{cases}
                \mathbf{0}, \mathrm{if ~infeas} = 0 \\
                - (\mathbf{c}_{\mathbf{x}_t}  + \mathbf{c}_{\mathbf{u}_t} \mathbf{K}_t), \mathrm{if ~infeas} = 1 \\
                                \end{cases}
                                 \\
                \mathbf{k}_t^{\mathbf{y}} &= \begin{cases}
                \mathbf{0}, \mathrm{if ~infeas} = 0 \\
                - (\mathbf{c}_{t} + \mathbf{y}_{t}) - \mathbf{c}_{\mathbf{u}_t} \mathbf{k}_t, \mathrm{if ~infeas} = 1 \\
                                \end{cases}                 \\
                \mathbf{V}_t &= \mathbf{Q}_{\mathbf{x}_t, \mathbf{x}_t}
                    + \mathbf{Q}_{\mathbf{x}_t, \mathbf{u}_t}\mathbf{K}_t
                    + \mathbf{K}_t^\top\mathbf{Q}_{\mathbf{u}_t, \mathbf{x}_t}
                    + \mathbf{K}_t^\top\mathbf{Q}_{\mathbf{u}_t, \mathbf{u}_t}\mathbf{K}_t   \\
                \mathbf{v}_t &= \mathbf{q}_{\mathbf{x}_t}
                    + \mathbf{Q}_{\mathbf{x}_t, \mathbf{u}_t}\mathbf{k}_t
                    + \mathbf{K}_t^\top\mathbf{q}_{\mathbf{u}_t}
                    + \mathbf{K}_t^\top\mathbf{Q}_{\mathbf{u}_t, \mathbf{u}_t}\mathbf{k}_t   \\
            \end{align*}

      where :math:`\odot` denotes the tensor contraction and

        .. math::
            \begin{align*}
                \mathbf{V}_T &= p_{\mathbf{x}_T, \mathbf{x}_T} \\
                \mathbf{v}_T &= p_{\mathbf{x}_T} \\
                \mathbf{F}_t &= \begin{bmatrix} \mathbf{f}_{\mathbf{x}_t} \\
                                \mathbf{f}_{\mathbf{u}_t} \end{bmatrix} \\
                \mathbf{W}_t &= \begin{bmatrix} \mathbf{c}_{\mathbf{x}_t} \\
                                \mathbf{c}_{\mathbf{u}_t} \end{bmatrix} \\
                \mathbf{G}_t &= \begin{bmatrix} \mathbf{f}_{\mathbf{x}_t, \mathbf{x}_t}, \mathbf{f}_{\mathbf{x}_t, \mathbf{u}_t}\\
                                \mathbf{f}_{\mathbf{u}_t, \mathbf{x}_t}, \mathbf{f}_{\mathbf{u}_t, \mathbf{u}_t}\\  \end{bmatrix} \\
                \mathbf{S}_t &= \mathbf{diag}(\mathbf{s}_t) \\
                \mathbf{C}_t &= \mathbf{diag}(\mathbf{c}_t) \\
                \mathbf{r}_t &= \begin{cases}
                \mathbf{S}_t \mathbf{c}_t + \mu  \mathbf{1}, \mathrm{if ~infeas} = 0 \\
                \mathbf{S}_t \mathbf{y}_t - \mu  \mathbf{1}, \mathrm{if ~infeas} = 1 \\
                                \end{cases}  \\
                \hat{\mathbf{r}}_t &= \mathbf{S}_t(\mathbf{c}_t+\mathbf{y}_t) - \mathbf{r}_t
            \end{align*}

      As seen from the above equations, compared with LQR, additional terms involving
      :math:`\mathbf{W}` have been added into iteration, which are related to the inequality
      constraints. Additionally, feedback gains :math:`\mathbf{K}_t^{\mathbf{s}}`,
      :math:`\mathbf{k}_t^{\mathbf{s}}` for the dual variable are also introduced.

    - The forward recursion.

      For :math:`t = 0` to :math:`T-1`:

        .. math::
            \begin{align*}
                \mathbf{u}_t &= \mathbf{K}_t(\mathbf{x}_t - \mathbf{x}_t^{-}) + \mathbf{k}_t + \mathbf{u}_t^{-}\\
                \mathbf{s}_t &= \mathbf{K}_t^{\mathbf{s}}(\mathbf{x}_t - \mathbf{x}_t^{-}) + \mathbf{k}_t^{\mathbf{s}}  + \mathbf{s}_t^{-}\\
                \mathbf{y}_t &= \begin{cases}
                \mathbf{0}, \mathrm{if ~infeas} = 0 \\
                \mathbf{K}_t^{\mathbf{y}}(\mathbf{x}_t - \mathbf{x}_t^{-}) + \mathbf{k}_t^{\mathbf{y}}  + \mathbf{y}_t^{-}, \mathrm{if ~infeas} = 1 \\
                                \end{cases}  \\
                \mathbf{x}_{t+1} &= \mathbf{f}(\mathbf{x}_t,\mathbf{u}_t,t)  \\
            \end{align*}

      Additionally, some line-search and feasibility check algorithms are implemented therein.

    Then cost of the system over the time horizon:

        .. math::
            \mathbf{cost} \left( \mathbf{\tau}_t \right) = \sum\limits_{t=0}^{T-1}q(\mathbf{\tau}_t) + p(\mathbf{\tau}_T)

    Note:
        Some additional tricks were used in the implementation, i.e., egularization in the backwardpass,
        filter, line search in the forwardpass.

    From the learning perspective, this can be interpreted as a module with unknown parameters in
    :math:`\begin{Bmatrix} \mathbf{f}, \mathbf{c}, q, p \end{Bmatrix}`,
    which can be integrated into a larger end-to-end learning system.

    Note:
        The implementation is based on paper `Pavlov, Andrei, Iman Shames, and Chris Manzie.
        "Interior point differential dynamic programming." IEEE Transactions on Control Systems Technology 29.6 (2021): 2720-2727.
        <https://arxiv.org/pdf/2004.12710.pdf>`_.

    Example:
        >>> import pypose as pp
        >>> import torch as torch
        >>> import torch.nn as nn
        >>> from pypose.module.ipddp import IPDDP
        >>> from pypose.module.dynamics import NLS
        >>> device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        >>> class InvPend(NLS):
                def __init__(self, dt, length=[10.0], gravity=10.0):
                super(InvPend, self).__init__(xdim=2, udim=1, ydim=2)
                self.tau = dt
                self.length = length
                self.gravity = gravity
        >>> def state_transition(self, state, input, t=None):
                force = input.squeeze(-1)
                _dstate = torch.stack([state[...,1], force+self.gravity/self.length[0]*torch.sin(state[...,0].clone())], dim=-1)
                return state + torch.mul(_dstate, self.tau)
        >>> def observation(self, state, input, t=None):
                return state
        >>>
        >>> dt = 0.05   # Delta t
        >>> T = 5    # Number of time steps
        >>> state = torch.tensor([[-2.,0.], [-1., 0.], [-2.5, 1.]], device=device)
        >>>
        >>> sys = InvPend(dt)
        >>> ns, nc, n_batch = sys.xdim, sys.udim, state.shape[0]
        >>> input_all = 0.02*torch.ones(n_batch,  T,   nc, device=device)
        >>>
        >>> Q = torch.tile(dt*torch.eye(ns, ns, device=device), (n_batch, T, 1, 1))
        >>> R = torch.tile(dt*torch.eye(nc, nc, device=device), (n_batch, T, 1, 1))
        >>> S = torch.tile(torch.zeros(ns, nc, device=device), (n_batch, T, 1, 1))
        >>> c = torch.tile(torch.zeros(1, device=device), (n_batch, T))
        >>> stage_cost = pp.module.QuadCost(Q, R, S, c)
        >>> terminal_cost = pp.module.QuadCost(10./dt*Q[...,0:1,:,:], R[...,0:1,:,:], S[...,0:1,:,:], c[...,0:1]) # special stagecost with T=1
        >>>
        >>> gx = torch.tile(torch.zeros( 2*nc, ns, device=device), (n_batch, T, 1, 1))
        >>> gu = torch.tile(torch.vstack( (torch.eye(nc, nc, device=device), - torch.eye(nc, nc, device=device)) ), (n_batch, T, 1, 1))
        >>> g = torch.tile(torch.hstack( (-0.25 * torch.ones(nc, device=device), -0.25 * torch.ones(nc, device=device)) ), (n_batch, T, 1))
        >>> lincon = pp.module.LinCon(gx, gu, g)
        >>>
        >>> traj_opt = [None for batch_id in range(n_batch)]
        >>> for batch_id in range(n_batch): # use for loop and keep the IPDDP
        >>>     stage_cost = pp.module.QuadCost(Q[batch_id:batch_id+1], R[batch_id:batch_id+1], S[batch_id:batch_id+1], c[batch_id:batch_id+1])
        >>>     terminal_cost = pp.module.QuadCost(10./dt*Q[batch_id:batch_id+1,0:1,:,:], R[batch_id:batch_id+1,0:1,:,:], S[batch_id:batch_id+1,0:1,:,:], c[batch_id:batch_id+1,0:1])
        >>>     lincon = pp.module.LinCon(gx[batch_id:batch_id+1], gu[batch_id:batch_id+1], g[batch_id:batch_id+1])
        >>>     ipddp = IPDDP(sys, stage_cost, terminal_cost, lincon, T, B=(1,))
        >>>     x_init, u_init = state[batch_id:batch_id+1], input_all[batch_id:batch_id+1]
        >>>     traj_opt[batch_id] = ipddp.solver(x_init, u_init=None, verbose=True)


    '''
    def __init__(self, sys, stage_cost, terminal_cost, cons, T, B = (1,)):
        super().__init__()
        self.f_fn, self.p_fn, self.q_fn, self.c_fn = sys, terminal_cost, stage_cost, cons
        self.constraint_flag, self.contraction_flag = True, True
        ns, nc, ncons, self.T = self.f_fn.xdim, self.f_fn.udim, self.c_fn.cdim, T
        self.dargs = {'dtype': torch.float64, 'device': torch.device("cuda" if torch.cuda.is_available() else "cpu")}

        #==============algorithm parameter===================
        self.mu, self.maxiter, self.tol, self.infeas = 1.0, 50, torch.tensor([1.0e-7], **self.dargs), False
        #====================================================

        # ========== quantities in forward pass =============
        self.x, self.u = torch.zeros(B + (self.T + 1, ns), **self.dargs), torch.zeros(B + (self.T, nc), **self.dargs)
        self.c, self.s, self.y = torch.zeros(B + (self.T, ncons), **self.dargs), 0.1 * torch.ones(B + (self.T, ncons), **self.dargs), 0.01 * torch.ones(B + (self.T, ncons), **self.dargs)
        # s is dual variable for inequality constraint c, y is slack variable for infeasible case

        #---------------derivatives----------
        # terms related with system dynamics
        self.fx, self.fu = torch.zeros(B + (self.T, ns, ns), **self.dargs), torch.zeros(B + (self.T, ns, nc), **self.dargs)
        self.fxx, self.fxu, self.fuu = torch.zeros(B + (self.T, ns, ns, ns), **self.dargs), torch.zeros(B + (self.T, ns, ns, nc), **self.dargs), torch.zeros(B + (self.T, ns, nc, nc), **self.dargs)
        # terms related with stage cost
        self.qx, self.qu = torch.zeros(B + (self.T, ns), **self.dargs), torch.zeros(B + (self.T, nc), **self.dargs)
        self.qxx, self.qxu, self.quu = torch.zeros(B + (self.T, ns, ns), **self.dargs), torch.zeros(B + (self.T, ns, nc), **self.dargs), torch.zeros(B + (self.T, nc, nc), **self.dargs)
        # terms related with terminal cost
        self.px, self.pxx = torch.zeros(B + (ns,), **self.dargs), torch.zeros(B + (ns, ns), **self.dargs)
        # terms related with constraint
        self.cx, self.cu = torch.zeros(B + (self.T, ncons, ns), **self.dargs), torch.zeros(B + (self.T, ncons, nc), **self.dargs)
        #------------------------------------
        self.filter = torch.Tensor([[torch.inf], [0.]])
        self.err, self.cost, self.logcost = torch.zeros(B, **self.dargs), torch.zeros(B, **self.dargs), torch.zeros(B, **self.dargs)
        self.step, self.fp_failed, self.stepsize, self.reg_exp_base = 0, False, 1.0, 1.6
        #===================================================

        #=========quantities used in backwardpass===========
        self.ky, self.Ky = torch.zeros(B + (self.T, ncons), **self.dargs), torch.zeros(B + (self.T, ncons, ns), **self.dargs)
        self.ks, self.Ks = torch.zeros(B + (self.T, ncons), **self.dargs), torch.zeros(B + (self.T, ncons, ns), **self.dargs)
        self.ku, self.Ku = torch.zeros(B + (self.T, nc), **self.dargs), torch.zeros(B + (self.T, nc, ns), **self.dargs)
        self.opterr, self.reg, self.bp_failed, self.recovery = torch.tensor([0.], **self.dargs), torch.tensor([0.], **self.dargs), False, 0
        #===================================================

    def getDerivatives(self):
        self.p_fn.set_refpoint(self.x[...,-1,:], self.u[...,-1,:])
        self.px, self.pxx = self.p_fn.cx, self.p_fn.cxx.squeeze(0).squeeze(1)

        for t in range(self.T):
            self.f_fn.set_refpoint(self.x[...,t,:], self.u[...,t,:])
            self.fx[...,t,:,:], self.fu[...,t,:,:] = self.f_fn.A.squeeze(0).squeeze(1), self.f_fn.B.squeeze(0).squeeze(1)
            self.fxx[...,t,:,:,:], self.fxu[...,t,:,:,:], self.fuu[...,t,:,:,:] = self.f_fn.fxx.squeeze(0).squeeze(1).squeeze(2), self.f_fn.fxu.squeeze(0).squeeze(1).squeeze(2), self.f_fn.fuu.squeeze(0).squeeze(1).squeeze(2)

        self.q_fn.set_refpoint(self.x[...,:-1,:], self.u)
        self.qx, self.qu = self.q_fn.cx, self.q_fn.cu
        self.qxx, self.qxu, self.quu = self.q_fn.cxx, self.q_fn.cxu, self.q_fn.cuu
        # squeezed inside cxx definition

        self.c_fn.set_refpoint(self.x[...,:-1,:], self.u)
        self.c, self.cx, self.cu = self.c_fn(self.x[...,:-1,:], self.u), self.c_fn.gx, self.c_fn.gu

        self.Q = torch.cat([torch.cat([self.qxx, self.qxu],dim=-1), torch.cat([self.qxu.mT, self.quu],dim=-1)], dim=-2)
        self.G =  torch.cat([torch.cat([self.fxx, self.fxu],dim=-1), torch.cat([self.fxu.mT, self.fuu],dim=-1)], dim=-2)
        self.p, self.W, self.F = torch.cat([self.qx, self.qu],dim=-1), torch.cat([self.cx, self.cu],dim=-1), torch.cat([self.fx, self.fu],dim=-1)

    def resetfilter(self):
        if self.infeas:
            self.logcost = self.cost - self.mu * self.y.log().sum(-1).sum(-1)
            self.err =  torch.linalg.norm(self.c + self.y, dim=-1).sum(-1)
            if self.err < self.tol: self.err = torch.zeros(self.x.shape[:-2], dtype=self.x.dtype, device=self.x.device)
        else:
            self.logcost = self.cost - self.mu * (-self.c).log().sum(-1).sum(-1)
            self.err = torch.zeros(self.x.shape[:-2], dtype=self.x.dtype, device=self.x.device)

        self.filter = torch.stack((self.logcost, self.err), dim=-1).unsqueeze(-2)
        self.step, self.failed = 0, False

    def backwardpasscompact(self, lastIterFlag=False):
        r'''
        Compute controller gains for next iteration from current trajectory.
        '''
        B = self.x.shape[:-2]
        if lastIterFlag:
            B = self.p.shape[:-2] # redefine B as last iteration consider batched version
            ns, nc = self.Qx_terminal.size(-1), self.F.size(-1) - self.Qx_terminal.size(-1)
            self.Ku = torch.zeros(B + (self.T, nc, ns), dtype=self.p.dtype, device=self.p.device)
            self.ku = torch.zeros(B + (self.T, nc), dtype=self.p.dtype, device=self.p.device)

        ns = self.x.shape[-1]
        if not lastIterFlag:
            c_err, r_err, qu_err = torch.zeros(B, **self.dargs), torch.zeros(B, **self.dargs), torch.zeros(B, **self.dargs)

            # set regularization parameter
            if (self.fp_failed or self.bp_failed):
                self.reg += torch.tensor([1.], **self.dargs)
            elif (self.step == 0): # + int( self.fp_failed ....)
                self.reg -= torch.tensor([1.], **self.dargs)
            elif (self.step <= 3):
                self.reg = self.reg
            else:
                self.reg += torch.tensor([1.], **self.dargs)

            self.reg = torch.clamp(self.reg, torch.tensor([0.], **self.dargs), torch.tensor([24.], **self.dargs))

            V, v = self.pxx, self.px
        else:
            V, v = self.Qxx_terminal, self.Qx_terminal

        # --------- backward recursions ---------------------------
        # similar to iLQR backward recursion, but more variables involved
        for t in range(self.T-1, -1, -1):
            Ft = self.F[...,t,:,:]
            Qt = self.Q[...,t,:,:] + Ft.mT @ V @ Ft
            qt = self.p[...,t,:] + bmv(Ft.mT, v)
            if self.contraction_flag:
                Qt += btdot(v, self.G[...,t,:,:,:]) # terms related to 2nd-order derivatives of dynamics
            if self.constraint_flag:
                qt += bmv(self.W[...,t,:,:].mT, self.s[...,t,:]) # terms related to 2nd-order derivatives of constraints

            if (self.infeas): #  start from infeasible/feasible trajs.
                Wt = self.W[...,t,:,:]
                st, ct, yt = self.s[...,t,:], self.c[...,t,:], self.y[...,t,:]
                r = st * yt - self.mu
                rhat, yinv = st * (ct + yt) - r, 1. / yt
                SYinv = torch.diag_embed(st * yinv)

                Qt += Wt.mT @ SYinv @ Wt
                qt += bmv(Wt.mT, yinv * rhat)
                Qxx, Qxu = Qt[...,:ns,:ns], Qt[...,:ns,ns:]
                Qux, Quu = Qt[...,ns:,:ns], Qt[...,ns:,ns:]
                qx, qu = qt[...,:ns], qt[...,ns:]
                Quu_reg = Quu + self.Q[...,t,ns:,ns:] * (pow(self.reg_exp_base, self.reg) - 1.)

                try:
                    lltofQuuReg = torch.linalg.cholesky(Quu_reg)
                except:
                    self.bp_failed, self.opterr = True, torch.inf

                Quu_reg_inv = torch.linalg.pinv(Quu_reg)
                self.Ku[...,t,:,:] = Kut = - Quu_reg_inv @ Qux
                self.ku[...,t,:] = kut = - bmv(Quu_reg_inv, qu)

                cx, cu = Wt[..., :ns], Wt[..., ns:]
                self.ks[...,t,:] = yinv * (rhat + st * bmv(cu, kut))
                self.Ks[...,t,:,:] = SYinv @ (cx + cu @ Kut)
                self.ky[...,t,:] = - (ct + yt) - bmv(cu, kut)
                self.Ky[...,t,:,:] = - (cx + cu @ Kut)
            else:
                Wt, st, ct = self.W[...,t,:,:], self.s[...,t,:], self.c[...,t,:]
                r, cinv = st * ct + self.mu, 1. / ct
                SCinv = torch.diag_embed(st * cinv)

                Qt -= Wt.mT @ SCinv @ Wt
                qt -= bmv(Wt.mT, cinv * r)
                Qxx, Qxu = Qt[...,:ns,:ns], Qt[...,:ns,ns:]
                Qux, Quu = Qt[...,ns:,:ns], Qt[...,ns:,ns:]
                qx, qu = qt[...,:ns], qt[..., ns:]

                if not lastIterFlag:
                    Quu_reg = Quu + self.Q[...,t,ns:,ns:] * (pow(self.reg_exp_base, self.reg) - 1.)
                    try:
                        lltofQuuReg = torch.linalg.cholesky(Quu_reg)
                    except:
                        self.bp_failed, self.opterr = True, torch.inf
                else:
                    Quu_reg = Quu

                Quu_reg_inv = torch.linalg.pinv(Quu_reg)
                self.Ku[...,t,:,:] = Kut = - Quu_reg_inv @ Qux
                self.ku[...,t,:] = kut = - bmv(Quu_reg_inv, qu)

                if not lastIterFlag:
                    cx, cu = Wt[...,:ns], Wt[...,ns:]
                    self.ks[...,t,:] = - cinv * (r + st * bmv(cu, kut))
                    self.Ks[...,t,:,:] = - SCinv @ (cx + cu @ Kut)
                    self.ky[...,t,:] = torch.zeros(ct.shape[-1]) # dummy as zero
                    self.Ky[...,t,:,:] = torch.zeros(ct.shape[-1], ns)

            V = Qxx + Qxu @ Kut + Kut.mT @ Qux + Kut.mT @ Quu @ Kut
            v = qx  + bmv(Qxu, kut) + bmv(Kut.mT, qu) + bmv(Kut.mT @ Quu, kut)

            if not lastIterFlag:
                qu_err = torch.maximum(qu_err, torch.linalg.vector_norm(qu, float('inf'), dim=-1)  )
                r_err = torch.maximum(r_err, torch.linalg.vector_norm(r,  float('inf'), dim=-1)  )
                if self.infeas: c_err=torch.maximum(c_err, torch.linalg.vector_norm(ct + yt, float('inf'), dim=-1) )
        if not lastIterFlag:
            self.bp_failed, self.opterr = False, torch.maximum(torch.maximum(qu_err, c_err), r_err)

    def forwardpasscompact(self, lastIterFlag=False):
        r'''
        Compute new trajectory from controller gains.
        '''
        xold, uold, yold, sold, cold = self.x, self.u, self.y, self.s, self.c
        if not lastIterFlag:
            tau, steplist = torch.maximum(1.-self.mu,torch.tensor([0.99], **self.dargs)), pow(2.0, torch.linspace(-10, 0, 11, **self.dargs).flip(0))
            B = self.x.shape[:-2]
        else:
            steplist = torch.ones((1), **self.dargs) # skip linesearch
            B = self.p.shape[:-2]
            xold, uold = self.xold, self.uold

        xnew, unew, ynew, snew, cnew = torch.zeros_like(xold), torch.zeros_like(uold), torch.zeros_like(yold), torch.zeros_like(sold), torch.zeros_like(cold)
        logcost, err = torch.zeros(B), torch.zeros(B)
        for step in range(steplist.shape[0]): # line search
            failed, stepsize = False, steplist[step]
            xnew[...,0,:] = xold[...,0,:]
            xnewt = xnew[...,0,:]
            if (self.infeas): #  start from infeasible/feasible trajs.
                for t in range(self.T):
                    Kut, kut = self.Ku[...,t,:,:], self.ku[...,t,:]
                    Kst, kst = self.Ks[...,t,:,:], self.ks[...,t,:]
                    Kyt, kyt = self.Ky[...,t,:,:], self.ky[...,t,:]

                    ynew[...,t,:] = ynewt = yold[...,t,:] + stepsize * kyt + bmv(Kyt, xnewt - xold[...,t,:])
                    snew[...,t,:] = snewt = sold[...,t,:] + stepsize * kst + bmv(Kst, xnewt - xold[...,t,:])

                    if ((ynewt < (1-tau)*yold[...,t,:]).any() or (snewt<(1-tau)*sold[...,t,:]).any()):
                        failed = True
                        break
                    unew[...,t,:] = unewt = uold[...,t,:] + stepsize * kut + bmv(Kut, xnewt - xold[...,t,:])
                    xnew[...,t+1,:] = xnewt = self.f_fn(xnewt, unewt)[0]
            else:
                for t in range(self.T): # forward recursions
                    Kut, kut = self.Ku[...,t,:,:], self.ku[...,t,:]
                    unew[...,t,:] = unewt = uold[...,t,:] + stepsize * kut + bmv(Kut, xnewt - xold[...,t,:])
                    if not lastIterFlag: # check feasibility of forward recursions
                        Kst, kst = self.Ks[...,t,:,:], self.ks[...,t,:]
                        snew[...,t,:] = snewt = sold[...,t,:] + stepsize * kst + bmv(Kst, xnewt - xold[...,t,:])
                        cnew[...,t,:] = cnewt = self.c_fn(xnew[...,:-1,:], unew)[...,t,:]
                        if ((cnewt > (1-tau) * cold[...,t,:]).any() or (snewt < (1-tau) * sold[...,t,:]).any()):
                            failed = True
                            break
                    xnew[...,t+1,:] = xnewt = self.f_fn(xnewt, unewt)[0]

            if (failed):
                continue
            else:
                cost = self.q_fn(xnew[...,:-1,:], unew).sum(-1) + self.p_fn(xnew[...,-1,:],torch.zeros_like(unew[...,-1,:])).sum(-1)
                if not lastIterFlag:
                    if (self.infeas):
                        logcost = cost - self.mu * ynew.log().sum(-1).sum(-1)
                        cnew = self.c_fn(xnew[...,:-1,:], unew)
                        err = torch.linalg.vector_norm(cnew + ynew, 1, dim=-1).sum(-1)
                        err = torch.maximum(self.tol, err)
                    else:
                        logcost = cost - self.mu * (-cnew).log().sum(-1).sum(-1)
                        err = torch.zeros(B, **self.dargs)
                    # step filter, ref to R. Fletcher and S. Leyffer, “Nonlinear programming without a penalty function,” Mathematical programming, vol. 91, no. 2, pp. 239–269, 2002.
                    candidate = torch.stack((logcost, err), dim=-1)
                    if torch.any( torch.all(candidate-torch.tile(torch.tensor([1e-13, 0.], **self.dargs), B + (1,))>=self.filter, -1) ):
                        # relax a bit for numerical stability, strange; todo: any for each sample in a batch?
                        failed=True
                        continue
                    else:
                        idx = torch.all(candidate<=self.filter,-1)
                        self.filter = self.filter[torch.logical_not(idx)]
                        if self.filter.ndim <= 2:
                            self.filter = self.filter.unsqueeze(0)
                        self.filter=torch.cat((self.filter, candidate.unsqueeze(-2)), dim=-2)
                        break

        if (failed):
            self.stepsize, self.failed= 0.0, failed
        else:
            self.x, self.u, self.y, self.s, self.c = xnew, unew, ynew, snew, cnew
            self.cost, self.err, self.stepsize, self.step, self.failed = cost, err, stepsize, step, False

    def forward(self, x_init, u_init=None, verbose=False):
        r'''
        Call forwardpass and backwardpass to solve an optimal trajectory for general nonlinear system
        '''
        with torch.no_grad():
            time_start = time.time()

            self.x[...,0,:] = x_init
            if u_init is not None: self.u = u_init
            for t in range(self.T):
                self.x[...,t+1,:], _ = self.f_fn(self.x[...,t,:],self.u[...,t,:])
            self.c = self.c_fn(self.x[...,:-1,:], self.u)
            if (self.c > 0).any(): self.infeas = True
            self.cost = self.q_fn(self.x[...,:-1,:], self.u).sum(-1) \
                        + self.p_fn(self.x[...,-1,:],torch.zeros_like(self.u[...,-1,:])).sum(-1)
            self.mu = self.cost / self.T / self.s[...,0,:].shape[-1]
            self.resetfilter()
            self.reg, self.bp_failed, self.recovery = 0.0, False, 0

            for iter in range(self.maxiter):
                while True:
                    # recompute the first, second derivatives of the updated trajectory
                    if not self.fp_failed: self.getDerivatives()
                    self.backwardpasscompact()
                    if not self.bp_failed:
                        break

                self.forwardpasscompact()
                time_used = time.time() - time_start
                if verbose:
                    if (iter % 5 == 1):
                        print('\n')
                        print('Iteration','Time','mu','Cost','Opt. error','Reg. power','Stepsize')
                        print('\n')
                    print('%-12d%-12.4g%-12.4g%-12.4g%-12.4g%-12d%-12.3f\n'%(
                                iter, time_used, self.mu, self.cost, self.opterr, self.reg, self.stepsize))

                #-----------termination conditions---------------
                if (max(self.opterr, self.mu)<=self.tol):
                    print("~~~Optimality reached~~~")
                    break

                if (self.opterr <= 0.2*self.mu):
                    self.mu = max(self.tol/10.0, min(0.2*self.mu, pow(self.mu, 1.2) ) )
                    self.resetfilter()
                    self.reg, self.bp_failed, self.recovery = 0.0, False, 0

                if iter == self.maxiter - 1:
                    print("max iter", self.maxiter, "reached, not the optimal one!")
                #-----------------------------------------------

        # Generate the trajectory with computational graph.
        # batched version has been removed temporarily, although this step can use batch computation
        with torch.autograd.set_detect_anomaly(True): # for debug
            with torch.no_grad(): # detach
                self.Qxx_terminal = self.pxx
                self.Qx_terminal = self.px
                self.Q = torch.cat( [torch.cat([self.qxx, self.qxu],dim=-1),
                                                torch.cat([self.qxu.mT, self.quu],dim=-1)], dim=-2)

                self.p = torch.cat([self.qx, self.qu],dim=-1)
                self.W = torch.cat([self.cx, self.cu],dim=-1)

                self.F = torch.cat([self.fx, self.fu],dim=-1)
                self.G = torch.cat([torch.cat([self.fxx, self.fxu],dim=-1),
                                    torch.cat([self.fxu.mT, self.fuu],dim=-1)], dim=-2)

                self.T = self.F.size(-3)
                self.xold = self.x
                self.uold = self.u

            # run backward and forward once
            self.infeas = False
            self.backwardpasscompact(lastIterFlag=True)
            self.forwardpasscompact(lastIterFlag=True)
        return self.x, self.u, self.cost
