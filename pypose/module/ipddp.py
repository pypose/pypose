import time
import torch as torch
import torch.nn as nn
from .. import bmv, bvmv, btdot

class IPDDP(nn.Module):
    r'''
    Interior-point differential dynamic programming.

    Args:
        sys (:obj:`instance`): System dynamics of the optimal control problem.
        stage_cost (:obj:`instance`): Stage cost of the optimal control problem.
        terminal_cost (:obj:`instance`): Terminal cost of the optimal control problem.
        cons (:obj:`instance`): Constraints of the optimal control problem.
        n_cons (:obj:`int`): Dimension of constraints.
        init_traj (:obj:`Dict`): Initial system trajectory.

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

    The IPDDP process can be summarised as iterative backward and forward recursions.

    - The backward recursion.

      For :math:`t` = :math:`T-1` to 0:

        .. math::
            \begin{align*}
                \mathbf{Q}_t &= \mathbf{Q}_t + \mathbf{F}_t^\top\mathbf{V}_{t+1}\mathbf{F}_t
                                    + \mathbf{v}_{t+1} \odot \mathbf{G}_t
                                    - \mathbf{W}_t^\top \mathbf{S}_t \mathbf{C}_t^{-1} \mathbf{W}_t   \\
                \mathbf{q}_t &= \mathbf{q}_t + \mathbf{F}_t^\top\mathbf{v}_{t+1}
                                          + \mathbf{W}_t^\top\mathbf{s}_t
                                          - \mathbf{W}_t^\top \mathbf{C}_t^{-1}\mathbf{r}_t\\
                \mathbf{K}_t &= -\mathbf{Q}_{\mathbf{u}_t, \mathbf{u}_t}^{-1}
                                                     \mathbf{Q}_{\mathbf{u}_t, \mathbf{x}_t} \\
                \mathbf{k}_t &= -\mathbf{Q}_{\mathbf{u}_t, \mathbf{u}_t}^{-1}
                                                           \mathbf{q}_{\mathbf{u}_t}         \\
                \mathbf{K}_t^{\mathbf{s}} &= - \mathbf{S}_t\mathbf{C}_t^{-1} (\mathbf{c}_{\mathbf{x}_t}
                                        + \mathbf{c}_{\mathbf{u}_t} \mathbf{K}_t) \\
                \mathbf{k}_t^{\mathbf{s}} &= - \mathbf{C}_t^{-1} (\mathbf{r}_t + \mathbf{S}_t
                                                    \mathbf{c}_{\mathbf{u}_t} \mathbf{k}_t) \\
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
                \mathbf{r}_t &= \mathbf{S}_t \mathbf{c}_t + \mu  \mathbf{1}  \\
            \end{align*}

      As seen from the above equations, compared with LQR, additional terms involving
      :math:`\mathbf{W}` have been added into iteration, which are related to the inequality
      constraints. Additionally, feedback gains :math:`\mathbf{K}_t^{\mathbf{s}}`,
      :math:`\mathbf{k}_t^{\mathbf{s}}` for the dual variable are also introduced.

    - The forward recursion.

      For :math:`t` = 0 to :math:`T-1`:

        .. math::
            \begin{align*}
                \mathbf{u}_t &= \mathbf{K}_t(\mathbf{x}_t - \mathbf{x}_t^{-}) + \mathbf{k}_t + \mathbf{u}_t^{-}\\
                \mathbf{s}_t &= \mathbf{K}_t^{\mathbf{s}}(\mathbf{x}_t - \mathbf{x}_t^{-}) + \mathbf{k}_t^{\mathbf{s}}  + \mathbf{s}_t^{-}\\
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
        The implementation is based on paper `Interior Point Differential Dynamic Programming
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
                super(InvPend, self).__init__()
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
        >>> ns, nc = 2, 1
        >>> n_batch = 3
        >>> state_all =      torch.zeros(n_batch, T+1, ns, device=device)
        >>> input_all = 0.02*torch.ones(n_batch,  T,   nc, device=device)
        >>> state_all[...,0,:] = state
        >>> init_traj = {'state': state_all, 'input': input_all}
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
        >>>     init_traj_sample = {'state': init_traj['state'][batch_id:batch_id+1], 'input': init_traj['input'][batch_id:batch_id+1]}
        >>>     solver = IPDDP(sys, stage_cost, terminal_cost, lincon, gx.shape[-2], init_traj_sample)
        >>>     traj_opt[batch_id] = solver.optimizer()

    '''
    def __init__(self, sys=None, stage_cost=None, terminal_cost=None, cons=None, n_cons=0, init_traj=None):
        super().__init__()
        self.f_fn = sys
        self.p_fn = terminal_cost
        self.q_fn = stage_cost
        self.c_fn = cons

        self.constraint_flag = True
        self.contraction_flag = True

        self.x, self.u = init_traj['state'], init_traj['input']
        B = self.x.shape[:-2]
        ns, nc, ncons, self.T = self.x.size(-1), self.u.size(-1), n_cons, self.u.size(-2)

        # algorithm parameter
        self.mu, self.maxiter, self.tol, self.infeas = 1.0, 50, torch.tensor([1.0e-7], dtype=self.x.dtype, device=self.x.device), False

        # quantities in forward pass
        self.c = torch.zeros(B + (self.T, ncons), dtype=self.x.dtype, device=self.x.device)
        self.s = 0.1 * torch.ones(B + (self.T, ncons), dtype=self.x.dtype, device=self.x.device) # dual variable for inequality constraint c
        self.y = 0.01 * torch.ones(B + (self.T, ncons), dtype=self.x.dtype, device=self.x.device) # slack variable for infeasible case

        # -------- derivatives --------------------------
        # terms related with system dynamics
        self.fx = torch.zeros(B + (self.T, ns, ns), dtype=self.x.dtype, device=self.x.device)
        self.fu = torch.zeros(B + (self.T, ns, nc), dtype=self.x.dtype, device=self.x.device)
        self.fxx = torch.zeros(B + (self.T, ns, ns, ns), dtype=self.x.dtype, device=self.x.device)
        self.fxu = torch.zeros(B + (self.T, ns, ns, nc), dtype=self.x.dtype, device=self.x.device)
        self.fuu = torch.zeros(B + (self.T, ns, nc, nc), dtype=self.x.dtype, device=self.x.device)
        # terms related with stage cost
        self.qx = torch.zeros(B + (self.T, ns), dtype=self.x.dtype, device=self.x.device)
        self.qu = torch.zeros(B + (self.T, nc), dtype=self.x.dtype, device=self.x.device)
        self.qxx = torch.zeros(B + (self.T, ns, ns), dtype=self.x.dtype, device=self.x.device)
        self.qxu = torch.zeros(B + (self.T, ns, nc), dtype=self.x.dtype, device=self.x.device)
        self.quu = torch.zeros(B + (self.T, nc, nc), dtype=self.x.dtype, device=self.x.device)
        # terms related with terminal cost
        self.px = torch.zeros(B + (ns,), dtype=self.x.dtype, device=self.x.device)
        self.pxx = torch.zeros(B + (ns, ns), dtype=self.x.dtype, device=self.x.device)
        # terms related with constraint
        self.cx = torch.zeros(B + (self.T, ncons, ns), dtype=self.x.dtype, device=self.x.device)
        self.cu = torch.zeros(B + (self.T, ncons, nc), dtype=self.x.dtype, device=self.x.device)
        # -----------------------------------------------

        self.filter = torch.Tensor([[torch.inf], [0.]])
        self.err, self.cost, self.logcost = torch.zeros(B, dtype=self.x.dtype, device=self.x.device), torch.zeros(B, dtype=self.x.dtype, device=self.x.device), torch.zeros(B, dtype=self.x.dtype, device=self.x.device)
        self.step, self.fp_failed, self.stepsize, self.reg_exp_base = 0, False, 1.0, 1.6

        # quantities used in backwardpass
        self.ky = torch.zeros(B + (self.T, ncons), dtype=self.x.dtype, device=self.x.device)
        self.Ky = torch.zeros(B + (self.T, ncons, ns), dtype=self.x.dtype, device=self.x.device)
        self.ks = torch.zeros(B + (self.T, ncons), dtype=self.x.dtype, device=self.x.device)
        self.Ks = torch.zeros(B + (self.T, ncons, ns), dtype=self.x.dtype, device=self.x.device)
        self.ku = torch.zeros(B + (self.T, nc), dtype=self.x.dtype, device=self.x.device)
        self.Ku = torch.zeros(B + (self.T, nc, ns), dtype=self.x.dtype, device=self.x.device)
        self.opterr, self.reg, self.bp_failed, self.recovery = 0., 0., False, 0

    def getDerivatives(self):
        self.p_fn.set_refpoint(self.x[...,-1,:], self.u[...,-1,:])
        self.px = self.p_fn.cx
        self.pxx = self.p_fn.cxx.squeeze(0).squeeze(1)

        for t in range(self.T):
            self.f_fn.set_refpoint(self.x[...,t,:], self.u[...,t,:])
            self.fx[...,t,:,:] = self.f_fn.A.squeeze(0).squeeze(1)
            self.fu[...,t,:,:] = self.f_fn.B.squeeze(0).squeeze(1)
            self.fxx[...,t,:,:,:] = self.f_fn.fxx.squeeze(0).squeeze(1).squeeze(2)
            self.fxu[...,t,:,:,:] = self.f_fn.fxu.squeeze(0).squeeze(1).squeeze(2)
            self.fuu[...,t,:,:,:] = self.f_fn.fuu.squeeze(0).squeeze(1).squeeze(2)

        self.q_fn.set_refpoint(self.x[...,:-1,:], self.u)
        self.qx = self.q_fn.cx
        self.qu = self.q_fn.cu
        self.qxx = self.q_fn.cxx # squeezed inside cxx definition
        self.qxu = self.q_fn.cxu
        self.quu = self.q_fn.cuu

        self.c = self.c_fn(self.x[...,:-1,:], self.u)
        self.c_fn.set_refpoint(self.x[...,:-1,:], self.u)
        self.cx = self.c_fn.gx
        self.cu = self.c_fn.gu

        self.Q = torch.cat([torch.cat([self.qxx, self.qxu],dim=-1),
                            torch.cat([self.qxu.mT, self.quu],dim=-1)], dim=-2)
        self.p  = torch.cat([self.qx, self.qu],dim=-1)
        self.W =  torch.cat([self.cx, self.cu],dim=-1)
        self.F =  torch.cat([self.fx, self.fu],dim=-1)
        self.G =  torch.cat([torch.cat([self.fxx, self.fxu],dim=-1),
                            torch.cat([self.fxu.mT, self.fuu],dim=-1)], dim=-2)

    def resetfilter(self):
        if (self.infeas):
            self.logcost = self.cost - self.mu * self.y.log().sum(-1).sum(-1)
            aa = torch.linalg.vector_norm(self.c + self.y, -1)
            self.err =  torch.linalg.norm(self.c + self.y, dim=-1).sum(-1)
            if (self.err < self.tol):
                self.err = torch.zeros(self.x.shape[:-2], dtype=self.x.dtype, device=self.x.device)
        else:
            self.logcost = self.cost - self.mu * (-self.c).log().sum(-1).sum(-1)
            self.err = torch.zeros(self.x.shape[:-2], dtype=self.x.dtype, device=self.x.device)

        self.filter = torch.stack((self.logcost, self.err), dim=-1).unsqueeze(-2)
        self.step = 0
        self.failed = False

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
            c_err, r_err, qu_err = torch.zeros(B, dtype=self.x.dtype, device=self.x.device), torch.zeros(B, dtype=self.x.dtype, device=self.x.device), torch.zeros(B, dtype=self.x.dtype, device=self.x.device)

            # set regularization parameter
            if (self.fp_failed or self.bp_failed):
                self.reg += 1.0
            elif (self.step == 0):
                self.reg -= 1.0
            elif (self.step <= 3):
                self.reg = self.reg
            else:
                self.reg += 1.0

            if (self.reg < 0.0):
                self.reg = 0.0
            elif (self.reg > 24.0):
                self.reg = 24.0

            # recompute the first, second derivatives of the updated trajectory
            if not self.fp_failed:
                self.getDerivatives()

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
                if (self.infeas):
                    c_err=torch.maximum(c_err, torch.linalg.vector_norm(ct + yt, float('inf'), dim=-1) )
        if not lastIterFlag:
            self.bp_failed, self.opterr = False, torch.maximum(torch.maximum(qu_err, c_err), r_err)

    def forwardpasscompact(self, lastIterFlag=False):
        r'''
        Compute new trajectory from controller gains.
        '''
        xold, uold, yold, sold, cold = self.x, self.u, self.y, self.s, self.c
        if not lastIterFlag:
            tau, steplist = torch.maximum(1.-self.mu,torch.tensor([0.99], dtype=self.x.dtype, device=self.x.device)), pow(2.0, torch.linspace(-10, 0, 11, dtype=self.x.dtype, device=self.x.device).flip(0))
            B = self.x.shape[:-2]
        else:
            steplist = torch.ones((1), dtype=self.x.dtype, device=self.x.device) # skip linesearch
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
                        err = torch.zeros(B, dtype=self.x.dtype, device=self.x.device)
                    # step filter, ref to R. Fletcher and S. Leyffer, “Nonlinear programming without a penalty function,” Mathematical programming, vol. 91, no. 2, pp. 239–269, 2002.
                    candidate = torch.stack((logcost, err), dim=-1)
                    if torch.any( torch.all(candidate-torch.tile(torch.tensor([1e-13, 0.], dtype=self.x.dtype, device=self.x.device), B + (1,))>=self.filter, -1) ):
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

    def solver(self, verbose=False):
        r'''
        Call forwardpass and backwardpass to solve an optimal trajectory for general nonlinear system
        '''
        time_start = time.time()

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
        return self

    def forward(self, fp_list):
        r'''
        Generate the trajectory with computational graph.

        Args:
            fp_list (List of :obj:`Class IPDDP`): The list of IPDDP class instances, each instance is solved from solver.

        Returns:
            Tensors (:obj:`Tensor`) including
            :math:`\mathbf{x}` (:obj:`Tensor`): The solved state sequence.
            :math:`\mathbf{u}` (:obj:`Tensor`): The solved input sequence.
            :math:`\mathbf{cost}` (:obj:`Tensor`):  The associated costs over the time horizon.
        '''
        with torch.autograd.set_detect_anomaly(True): # for debug
            # collect fp_list into batch
            n_batch = len(fp_list)
            with torch.no_grad(): # detach
                self.c, self.s = torch.cat([fp_list[batch_id].c for batch_id in range(n_batch)],dim=0), \
                                 torch.cat([fp_list[batch_id].s for batch_id in range(n_batch)],dim=0)
                self.Qxx_terminal = torch.cat([fp_list[batch_id].pxx for batch_id in range(n_batch)],dim=0)
                self.Qx_terminal = torch.cat([fp_list[batch_id].px for batch_id in range(n_batch)],dim=0)
                self.Q = torch.cat([torch.cat( [torch.cat([fp_list[batch_id].qxx,    fp_list[batch_id].qxu],dim=-1),
                                                torch.cat([fp_list[batch_id].qxu.mT, fp_list[batch_id].quu],dim=-1)], dim=-2)
                                    for batch_id in range(n_batch)], dim=0)
                self.p = torch.cat([ torch.cat([fp_list[batch_id].qx, fp_list[batch_id].qu],dim=-1)
                                                for batch_id in range(n_batch)], dim=0)
                self.W = torch.cat([ torch.cat([fp_list[batch_id].cx, fp_list[batch_id].cu],dim=-1)
                                                for batch_id in range(n_batch)], dim=0)
                self.F = torch.cat([ torch.cat([fp_list[batch_id].fx, fp_list[batch_id].fu],dim=-1)
                                                for batch_id in range(n_batch)], dim=0)
                self.G = torch.cat([ torch.cat([torch.cat([fp_list[batch_id].fxx, fp_list[batch_id].fxu],dim=-1),
                                                torch.cat([fp_list[batch_id].fxu.mT, fp_list[batch_id].fuu],dim=-1)], dim=-2)
                                    for batch_id in range(n_batch)], dim=0)

                self.T = self.F.size(-3)
                self.mu = torch.stack([fp_list[batch_id].mu for batch_id in range(n_batch)], dim=0) # use different mu for each sample
                self.xold = torch.cat([fp_list[batch_id].x for batch_id in range(n_batch)], dim=0)
                self.uold = torch.cat([fp_list[batch_id].u for batch_id in range(n_batch)], dim=0)

            # run backward and forward once
            self.infeas = False
            self.backwardpasscompact(lastIterFlag=True)
            self.forwardpasscompact(lastIterFlag=True)
        return self.x, self.u, self.cost
