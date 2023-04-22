import time
import torch as torch
import torch.nn as nn
from ..basics import bmv, bvmv, btdot

# class algParam:
#     r'''
#     The class of algorithm parameter.
#     '''
#     def __init__(self, mu=1.0, maxiter=50, tol=1.0e-7, infeas=False):
#         self.mu = mu  
#         self.maxiter = maxiter
#         self.tol = torch.tensor(tol)
#         self.infeas = infeas

class fwdPass:
    r'''
    The class used in forwardpass 
    here forwardpass means compute trajectory from inputs
    (different from the forward in neural network) 
    '''
    def __init__(self,sys=None, stage_cost=None, terminal_cost=None, cons=None, n_state=1, n_input=1, n_cons=0, horizon=1, init_traj=None):
        self.f_fn = sys
        self.p_fn = terminal_cost
        self.q_fn = stage_cost
        self.c_fn = cons
        self.T = horizon
        self.n_state = n_state
        self.n_input = n_input
        ncons = n_cons

        # initilize all the variables used in the forward pass
        # defined in dynamics function
        self.x = init_traj['state']
        self.u = init_traj['input']
        B = self.x.shape[:-2]
        ns, nc = self.x.size(-1), self.u.size(-1)
        self.c = torch.zeros(B + (self.T, ncons))
        self.y = 0.01 * torch.ones(B + (self.T, ncons))
        self.s = 0.1 * torch.ones(B + (self.T, ncons))
        self.mu = self.y * self.s

        # terms related with terminal cost
        self.px = torch.zeros(B + (ns,))
        self.pxx = torch.zeros(B + (ns, ns))

        # terms related with system dynamics
        self.fx = torch.zeros(B + (self.T, ns, ns))
        self.fu = torch.zeros(B + (self.T, ns, nc))
        self.fxx = torch.zeros(B + (self.T, ns, ns, ns))
        self.fxu = torch.zeros(B + (self.T, ns, ns, nc))
        self.fuu = torch.zeros(B + (self.T, ns, nc, nc))

        # terms related with stage cost
        self.qx = torch.zeros(B + (self.T, ns))
        self.qu = torch.zeros(B + (self.T, nc))
        self.qxx = torch.zeros(B + (self.T, ns, ns))
        self.qxu = torch.zeros(B + (self.T, ns, nc))
        self.quu = torch.zeros(B + (self.T, nc, nc))

        # terms related with constraint
        self.cx = torch.zeros(B + (self.T, ncons, ns))
        self.cu = torch.zeros(B + (self.T, ncons, nc))

        self.filter = torch.Tensor([[torch.inf], [0.]])
        self.err = 0.
        self.cost = 0.
        self.logcost = 0.
        self.step = 0
        self.failed = False
        self.stepsize = 1.0
        self.reg_exp_base = 1.6




class ddpOptimizer(nn.Module):
    r'''
    The class of ipddp optimizer
    iterates between forwardpass and backwardpass to get a final trajectory 
    '''
    # def __init__(self, system, constraint, Q, p, T):
    def __init__(self, sys=None, stage_cost=None, terminal_cost=None, cons=None, n_state=1, n_input=1, n_cons=0, horizon=None, init_traj=None):
        r'''
        Initialize three key classes
        '''
        super().__init__()
        # self.system = sys
        # self.constraint = cons
        # # self.T = T
        # # self.Q = Q
        # # self.p = p
        self.constraint_flag = True
        self.contraction_flag = True
        # # self.W = torch.randn(2, 5, 6, 7) # todo:change

        self.x, self.u = init_traj['state'], init_traj['input']
        B = self.x.shape[:-2]
        ns, nc, ncons, self.T = self.x.size(-1), self.u.size(-1), n_cons, horizon
        
        # self.alg = algParam()
        # self.fp = fwdPass(sys=sys, stage_cost=stage_cost, terminal_cost=terminal_cost, cons=cons, n_state=n_state, n_input=n_input, n_cons=n_cons, horizon=horizon, init_traj=init_traj)
        # self.bp = bwdPass(sys=sys,            cons=cons,n_state=n_state, n_input=n_input, n_cons=n_cons, horizon=horizon)
        
        # algorithm parameter
        self.mu, self.maxiter, self.tol, self.infeas = 1.0, 50, torch.tensor([1.0e-7]), False

        self.f_fn = sys
        self.p_fn = terminal_cost
        self.q_fn = stage_cost
        self.c_fn = cons

        # quantities in forward pass
        # defined in dynamics function
        self.c = torch.zeros(B + (self.T, ncons))
        self.y = 0.01 * torch.ones(B + (self.T, ncons))
        self.s = 0.1 * torch.ones(B + (self.T, ncons))
        self.mu = self.y * self.s
        # terms related with terminal cost
        self.px = torch.zeros(B + (ns,))
        self.pxx = torch.zeros(B + (ns, ns))
        # terms related with system dynamics
        self.fx = torch.zeros(B + (self.T, ns, ns))
        self.fu = torch.zeros(B + (self.T, ns, nc))
        self.fxx = torch.zeros(B + (self.T, ns, ns, ns))
        self.fxu = torch.zeros(B + (self.T, ns, ns, nc))
        self.fuu = torch.zeros(B + (self.T, ns, nc, nc))
        # terms related with stage cost
        self.qx = torch.zeros(B + (self.T, ns))
        self.qu = torch.zeros(B + (self.T, nc))
        self.qxx = torch.zeros(B + (self.T, ns, ns))
        self.qxu = torch.zeros(B + (self.T, ns, nc))
        self.quu = torch.zeros(B + (self.T, nc, nc))
        # terms related with constraint
        self.cx = torch.zeros(B + (self.T, ncons, ns))
        self.cu = torch.zeros(B + (self.T, ncons, nc))

        self.filter = torch.Tensor([[torch.inf], [0.]])
        self.err, self.cost, self.logcost = torch.zeros(B), torch.zeros(B), torch.zeros(B)
        self.step, self.fp_failed, self.stepsize, self.reg_exp_base = 0, False, 1.0, 1.6

        # quantities used in backward
        self.ky = torch.zeros(B + (self.T, ncons))
        self.Ky = torch.zeros(B + (self.T, ncons, ns))
        self.ks = torch.zeros(B + (self.T, ncons))
        self.Ks = torch.zeros(B + (self.T, ncons, ns))
        self.ku = torch.zeros(B + (self.T, nc))
        self.Ku = torch.zeros(B + (self.T, nc, ns))
        self.opterr, self.reg, self.bp_failed, self.recovery = 0., 0., False, 0

    def computeall(self): #todo merge this
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

    def resetfilter(self, infeas, mu):
        self.logcost, self.err = torch.zeros(self.x.shape[:-2]), torch.zeros(self.x.shape[:-2])
        if (infeas):
            for i in range(self.N): 
                self.logcost -= alg.mu * self.y[i].log().sum()
                self.err += torch.linalg.vector_norm(self.c[i]+self.y[i], 1)
            if (self.err < alg.tol):
                self.err = torch.Tensor([0.0])
        else:
            self.logcost = - mu * (-self.c).log().sum(-1).sum(-1)

        self.filter = torch.stack((self.logcost, self.err), dim=-1).unsqueeze(-2)
        self.step = 0
        self.failed = False

    def backwardpass(self):
        r'''
        Compute controller gains for next iteration from current trajectory.
        '''
        fp = self.fp
        bp = self.bp
        alg = self.alg

        self.n_state = fp.n_state
        self.n_input = fp.n_input
        self.N = fp.N
        dV = [0.0,0.0]
        c_err = torch.tensor(0.0)
        mu_err = torch.tensor(0.0)
        Qu_err = torch.tensor(0.0)

        # set regularization parameter
        if (fp.failed or bp.failed):
            bp.reg += 1.0
        elif (fp.step == 0):
            bp.reg -= 1.0
        elif (fp.step <= 3):
            bp.reg = bp.reg
        else:
            bp.reg += 1.0

        if (bp.reg < 0.0):
            bp.reg = 0.0
        elif (bp.reg > 24.0):
            bp.reg = 24.0

        # recompute the first, second derivatives of the updated trajectory
        if ~fp.failed:
            fp.computeall()
        
        x, u, c, y, s, mu = fp.x, fp.u, fp.c, fp.y, fp.s, fp.mu 
        Vx, Vxx = fp.px, fp.pxx
        fx,fu,fxx,fxu,fuu = fp.fx, fp.fu, fp.fxx, fp.fxu, fp.fuu
        qx,qu,qxx,qxu,quu = fp.qx, fp.qu, fp.qxx, fp.qxu, fp.quu   
        cx, cu = fp.cx, fp.cu

        # backward recursions, similar to iLQR backward recursion, but more variables involved
        for i in range(self.N-1, -1, -1):
            Qx = qx[i] + cx[i].mT.matmul(s[i]) + fx[i].mT.matmul(Vx)
            Qu = qu[i] + cu[i].mT.matmul(s[i]) + fu[i].mT.matmul(Vx) # (5b)

            fxiVxx = fx[i].mT.matmul(Vxx)
            Qxx = qxx[i] + fxiVxx.matmul(fx[i])  + torch.tensordot(Vx.mT,fxx[i],dims=1).squeeze(0)
            Qxu = qxu[i] + fxiVxx.matmul(fu[i])  + torch.tensordot(Vx.mT,fxu[i],dims=1).squeeze(0)
            Quu = quu[i] + fu[i].mT.matmul(Vxx).matmul(fu[i])  + torch.tensordot(Vx.mT,fuu[i],dims=1).squeeze(0)  # (5c-5e)
            Quu = 0.5 * (Quu + Quu.mT)
            Quu_reg = Quu + quu[i] * (pow(fp.reg_exp_base, bp.reg) - 1.)

            if (alg.infeas): #  start from infeasible/feasible trajs.
                r = s[i] * y[i] - alg.mu
                rhat = s[i] * (c[i] + y[i]) - r
                yinv = 1. / y[i]
                tempv1 = s[i] * yinv
                SYinv = torch.diag(tempv1.squeeze())
                cuitSYinvcui = cu[i].mT.matmul(SYinv).matmul(cu[i])
                SYinvcxi = SYinv.matmul(cx[i])

                try: 
                    lltofQuuReg = torch.linalg.cholesky(Quu_reg + cuitSYinvcui) # compute the Cholesky decomposition 
                except: 
                    bp.failed = True
                    bp.opterr = torch.inf 
                    self.fp = fp
                    self.bp = bp
                    self.alg = alg
                    return

                tempv2 = yinv * rhat
                Qu += cu[i].mT.matmul(tempv2)
                tempQux = Qxu.mT + cu[i].mT.matmul(SYinvcxi)
                tempm = torch.hstack( (Qu, tempQux) )

                kK = - torch.linalg.solve(Quu_reg + cuitSYinvcui, tempm)
                ku = torch.unsqueeze(kK[:,0],-1)
                Ku = kK[:,1:]
                cuiku = cu[i].matmul(ku)
                cxiPluscuiKu = cx[i] + cu[i].matmul(Ku)
                
                bp.ks[i] = yinv * (rhat + s[i] * cuiku)
                bp.Ks[i] = SYinv.matmul(cxiPluscuiKu)
                bp.ky[i] = - (c[i] + y[i]) - cuiku
                bp.Ky[i] = -cxiPluscuiKu

                Quu = Quu + cuitSYinvcui
                Qxu = tempQux.mT # Qxu + cx[i].transpose() * SYinvcui
                Qxx += cx[i].mT.matmul(SYinvcxi)
                Qx += cx[i].mT.matmul(tempv2)

            else:
                r = s[i] *  c[i] + alg.mu
                cinv = 1. / c[i]
                tempv1 = s[i] * cinv
                SCinv = torch.diag(tempv1.squeeze())
                SCinvcui = SCinv.matmul(cu[i])
                SCinvcxi = SCinv.matmul(cx[i])
                cuitSCinvcui = cu[i].mT.matmul(SCinvcui)
                
                try:
                    lltofQuuReg = torch.linalg.cholesky(Quu_reg - cuitSCinvcui) # compute the Cholesky decomposition 
                except: 
                    bp.failed = True
                    bp.opterr = torch.inf
                    self.fp = fp
                    self.bp = bp
                    self.alg = alg
                    return

                tempv2 = cinv * r
                Qu -= cu[i].mT.matmul(tempv2) # (12b)            
                tempQux = Qxu.mT - cu[i].mT.matmul(SCinvcxi)
                temp = torch.hstack(( Qu, tempQux))

                kK = - torch.linalg.solve(Quu_reg - cuitSCinvcui, temp)
                ku = torch.unsqueeze(kK[:,0],-1)
                Ku = kK[:,1:]

                cuiku = cu[i].matmul(ku)
                bp.ks[i] = - cinv * (r + s[i] * cuiku)
                bp.Ks[i] = - SCinv.matmul(cx[i] + cu[i].matmul(Ku)) # (11) checked
                bp.ky[i] = torch.zeros(c[i].shape[0], 1)
                bp.Ky[i] = torch.zeros(c[i].shape[0], self.n_state)       
                Quu = Quu - cuitSCinvcui # (12e)
                Qxu = tempQux.mT # Qxu - cx[i].transpose() * SCinvcui; // (12d)
                Qxx -= cx[i].mT.matmul(SCinvcxi) # (12c)
                Qx -= cx[i].mT.matmul(tempv2) # (12a)
            

            dV[0] += (ku.mT.matmul(Qu))

            QxuKu = Qxu.matmul(Ku)
            KutQuu = Ku.mT.matmul(Quu)

            dV[1] += 0.5 * ku.mT.matmul(Quu).matmul(ku)
            Vx = Qx + Ku.mT.matmul(Qu) + KutQuu.matmul(ku) + Qxu.matmul(ku) # (btw 11-12)
            Vxx = Qxx + QxuKu.mT + QxuKu + KutQuu.matmul(Ku) # (btw 11-12)
            Vxx = 0.5 * ( Vxx + Vxx.mT) # for symmetry

            bp.ku[i] = ku
            bp.Ku[i] = Ku

            Qu_err = torch.maximum(Qu_err, torch.linalg.vector_norm(Qu, float('inf'))  )
            mu_err = torch.maximum(mu_err, torch.linalg.vector_norm(r,  float('inf'))  )
            if (alg.infeas):
                c_err=torch.maximum(c_err, torch.linalg.vector_norm(c[i]+y[i], float('inf')) )

        bp.failed = False
        bp.opterr = torch.maximum( torch.maximum( Qu_err, c_err), mu_err)
        bp.dV = dV

        self.fp = fp
        self.bp = bp
        self.alg = alg

    def backwardpasscompact(self):
        r'''
        Compute controller gains for next iteration from current trajectory.
        '''
        ns = self.x.shape[-1]
        c_err, mu_err, qu_err = torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)

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
        if ~self.fp_failed:
            self.computeall()
        
        self.Q = torch.cat([torch.cat([self.qxx, self.qxu],dim=-1),
                            torch.cat([self.qxu.mT, self.quu],dim=-1)], dim=-2)                                     
        self.p  = torch.cat([self.qx, self.qu],dim=-1)
        self.W =  torch.cat([self.cx, self.cu],dim=-1)
        self.F =  torch.cat([self.fx, self.fu],dim=-1) 
        self.G =  torch.cat([torch.cat([self.fxx, self.fxu],dim=-1),
                            torch.cat([self.fxu.mT, self.fuu],dim=-1)], dim=-2) 
                            
        # backward recursions, similar to iLQR backward recursion, but more variables involved
        V, v = self.pxx, self.px
        for t in range(self.T-1, -1, -1):
            Ft = self.F[...,t,:,:]
            Qt = self.Q[...,t,:,:] + Ft.mT @ V @ Ft
            if self.contraction_flag: 
                Qt += btdot(v, self.G[...,t,:,:,:]) # todo :check!!!!
            qt = self.p[...,t,:] + bmv(Ft.mT, v) 
            if self.constraint_flag:
                qt += bmv(self.W[...,t,:,:].mT, self.s[...,t,:])


            if (self.infeas): #  start from infeasible/feasible trajs.
                r = s[i] * y[i] - alg.mu
                rhat = s[i] * (c[i] + y[i]) - r
                yinv = 1. / y[i]
                tempv1 = s[i] * yinv
                SYinv = torch.diag(tempv1.squeeze())
                cuitSYinvcui = cu[i].mT.matmul(SYinv).matmul(cu[i])
                SYinvcxi = SYinv.matmul(cx[i])

                try: 
                    lltofQuuReg = torch.linalg.cholesky(Quu_reg + cuitSYinvcui) # compute the Cholesky decomposition 
                except: 
                    bp.failed = True
                    bp.opterr = torch.inf 
                    self.fp = fp
                    self.bp = bp
                    self.alg = alg
                    return

                tempv2 = yinv * rhat
                Qu += cu[i].mT.matmul(tempv2)
                tempQux = Qxu.mT + cu[i].mT.matmul(SYinvcxi)
                tempm = torch.hstack( (Qu, tempQux) )

                kK = - torch.linalg.solve(Quu_reg + cuitSYinvcui, tempm)
                ku = torch.unsqueeze(kK[:,0],-1)
                Ku = kK[:,1:]
                cuiku = cu[i].matmul(ku)
                cxiPluscuiKu = cx[i] + cu[i].matmul(Ku)
                
                bp.ks[i] = yinv * (rhat + s[i] * cuiku)
                bp.Ks[i] = SYinv.matmul(cxiPluscuiKu)
                bp.ky[i] = - (c[i] + y[i]) - cuiku
                bp.Ky[i] = -cxiPluscuiKu

                Quu = Quu + cuitSYinvcui
                Qxu = tempQux.mT # Qxu + cx[i].transpose() * SYinvcui
                Qxx += cx[i].mT.matmul(SYinvcxi)
                Qx += cx[i].mT.matmul(tempv2)

            else:
                Wt = self.W[...,t,:,:]
                st, ct = self.s[...,t,:], self.c[...,t,:] 
                r = st *  ct + self.mu
                cinv = 1. / ct
                SCinv = torch.diag_embed(st * cinv)

                Qt += - Wt.mT @ SCinv @ Wt
                qt += - bmv(Wt.mT, cinv * r)
                Qxx, Qxu = Qt[..., :ns, :ns], Qt[..., :ns, ns:]
                Qux, Quu = Qt[..., ns:, :ns], Qt[..., ns:, ns:]
                qx, qu = qt[..., :ns], qt[..., ns:]
                
                Quu_reg = Quu + self.Q[...,t,ns:,ns:] * (pow(self.reg_exp_base, self.reg) - 1.)
                    
                try:
                    lltofQuuReg = torch.linalg.cholesky(Quu_reg) # compute the Cholesky decomposition 
                except: 
                    self.bp_failed, self.opterr = True, torch.inf

                Quu_reg_inv = torch.linalg.pinv(Quu_reg)
                self.Ku[...,t,:,:] = Kut = - Quu_reg_inv @ Qux
                self.ku[...,t,:] = kut = - bmv(Quu_reg_inv, qu)
                    
                cx, cu = Wt[..., :ns], Wt[..., ns:]
                self.ks[...,t,:] = - cinv * (r + st * bmv(cu, kut))
                self.Ks[...,t,:,:] = - SCinv @ (cx + cu @ Kut)
                self.ky[...,t,:] = torch.zeros(ct.shape[0]) # omitted
                self.Ky[...,t,:,:] = torch.zeros(ct.shape[0], ns)       

            V = Qxx + Qxu @ Kut + Kut.mT @ Qux + Kut.mT @ Quu @ Kut
            v = qx  + bmv(Qxu, kut) + bmv(Kut.mT, qu) + bmv(Kut.mT @ Quu, kut)

            qu_err = torch.maximum(qu_err, torch.linalg.vector_norm(qu, float('inf'))  )
            mu_err = torch.maximum(mu_err, torch.linalg.vector_norm(r,  float('inf'))  )
            # if (alg.infeas): 
                #todo
                # c_err=torch.maximum(c_err, torch.linalg.vector_norm(ct+yt, float('inf')) )

        self.bp_failed, self.opterr = False, torch.maximum( torch.maximum( qu_err, c_err), mu_err)

    def forwardpass(self):
        r'''
        Compute new trajectory from controller gains.
        '''
        fp = self.fp
        bp = self.bp
        alg = self.alg

        xold, uold, yold, sold, cold=fp.x, fp.u, fp.y, fp.s, fp.c
        xnew, unew, ynew, snew, cnew=torch.zeros_like(fp.x), torch.zeros_like(fp.u), torch.zeros_like(fp.y), torch.zeros_like(fp.s), torch.zeros_like(fp.c)
        cost, costq, logcost = torch.Tensor([0.]), torch.Tensor([0.]), torch.Tensor([0.])
        qnew = torch.zeros(self.N, 1)
        stepsize = 0.
        err = torch.Tensor([0.])
        tau = max(0.99, 1-alg.mu)
        steplist = pow(2.0, torch.linspace(-10, 0, 11).flip(0) )
        failed = False
        for step in range(steplist.shape[0]): # line search
            failed = False
            stepsize = steplist[step]
            xnew[0] = xold[0]
            if (alg.infeas): #  start from infeasible/feasible trajs. 
                for i in range(self.N):
                    ynew[i] = yold[i] + stepsize*bp.ky[i]+bp.Ky[i].matmul((xnew[i]-xold[i]).mT)
                    snew[i] = sold[i] + stepsize*bp.ks[i]+bp.Ks[i].matmul((xnew[i]-xold[i]).mT)

                    if (    (ynew[i]<(1-tau)*yold[i]).any() or 
                            (snew[i]<(1-tau)*sold[i]).any()   ): 
                        failed = True
                        break
                    
                    unew[i] = uold[i] + (stepsize*bp.ku[i]+bp.Ku[i].matmul((xnew[i]-xold[i]).mT)).mT
                    xnew[i+1] = fp.computenextx(xnew[i], unew[i])
            else:
                for i in range(self.N): # forward recuisions
                    snew[i] = sold[i] + stepsize*bp.ks[i]+bp.Ks[i].matmul((xnew[i]-xold[i]).mT)
                    unew[i] = uold[i] + (stepsize*bp.ku[i]+bp.Ku[i].matmul((xnew[i]-xold[i]).mT)).mT
                    cnew[i] = fp.computec(xnew[i], unew[i])

                    if (    (cnew[i]>(1-tau)*cold[i]).any() or  
                            (snew[i]<(1-tau)*sold[i]).any()   ): # check if the inequality holds, with some thresholds
                        failed = True
                        break
                    xnew[i+1] = fp.computenextx(xnew[i], unew[i])
                

        
            if (failed):
                continue
            else:
                for i in range(self.N):
                    qnew[i] = fp.computeq(xnew[i], unew[i])
                cost = qnew.sum() + fp.computep(xnew[-1])
                costq = qnew.sum()

                logcost = cost.detach()
                err = torch.Tensor([0.])          
                if (alg.infeas):
                    for i in range(self.N): 
                        logcost -= alg.mu * ynew[i].log().sum()
                        cnew[i] = fp.computec(xnew[i], unew[i])
                        err += torch.linalg.vector_norm(cnew[i]+ynew[i], 1)
                    err = torch.maximum(alg.tol, err)
                else:
                    for i in range(self.N):
                        cnew[i] = fp.computec(xnew[i], unew[i])
                        logcost -= alg.mu * (-cnew[i]).log().sum()
                    err=torch.Tensor([0.])
                
                # step filter
                candidate = torch.vstack((logcost, err))
                # if torch.any( torch.all(candidate>=fp.filter, 0) ):
                if torch.any( torch.all(candidate-torch.Tensor([[1e-13],[0.]])>=fp.filter, 0) ):
                    # relax a bit for numerical stability, strange
                    failed=True
                    continue                    
                else:
                    idx=torch.all(candidate<=fp.filter,0) 
                    fp.filter = fp.filter[:,~idx]
                    fp.filter=torch.hstack((fp.filter,candidate))
                    break
                  
        if (failed):
            fp.failed=failed
            fp.stepsize=0.0
        else:
            fp.cost, fp.costq, fp.logcost = cost, costq, logcost
            fp.x, fp.u, fp.y, fp.s, fp.c, fp.q = xnew, unew, ynew, snew, cnew, qnew 
            fp.err=err
            fp.stepsize=stepsize
            fp.step=step
            fp.failed=False

        self.fp = fp
        self.bp = bp
        self.alg = alg

    def forwardpasscompact(self):
        r'''
        Compute new trajectory from controller gains.
        '''
        B = self.x.shape[:-2]
        xold, uold, yold, sold, cold=self.x, self.u, self.y, self.s, self.c
        xnew, unew, ynew, snew, cnew=torch.zeros_like(self.x), torch.zeros_like(self.u), torch.zeros_like(self.y), torch.zeros_like(self.s), torch.zeros_like(self.c)
        logcost, err = torch.zeros(B), torch.zeros(B)       
        failed, tau, steplist = False, max(0.99, 1-self.mu), pow(2.0, torch.linspace(-10, 0, 11).flip(0))
        for step in range(steplist.shape[0]): # line search
            failed, stepsize = False, steplist[step]
            xnewt = xold[..., 0, :]
            if (self.infeas): #  start from infeasible/feasible trajs. 
                for i in range(self.N):
                    ynew[i] = yold[i] + stepsize*bp.ky[i]+bp.Ky[i].matmul((xnew[i]-xold[i]).mT)
                    snew[i] = sold[i] + stepsize*bp.ks[i]+bp.Ks[i].matmul((xnew[i]-xold[i]).mT)

                    if (    (ynew[i]<(1-tau)*yold[i]).any() or 
                            (snew[i]<(1-tau)*sold[i]).any()   ): 
                        failed = True
                        break
                    
                    unew[i] = uold[i] + (stepsize*bp.ku[i]+bp.Ku[i].matmul((xnew[i]-xold[i]).mT)).mT
                    xnew[i+1] = fp.computenextx(xnew[i], unew[i])
            else:
                for t in range(self.T): # forward recuisions
                    Kut, kut = self.Ku[..., t, :, :], self.ku[..., t, :]
                    Kst, kst = self.Ks[..., t, :, :], self.ks[..., t, :]
                    snew[..., t, :] = snewt = sold[..., t, :] + stepsize * kst + bmv(Kst, xnewt - xold[..., t, :])
                    unew[..., t, :] = unewt = uold[..., t, :] + stepsize * kut + bmv(Kut, xnewt - xold[..., t, :])
                    cnew[..., t, :] = cnewt = self.c_fn(xnew[..., :-1, :], unew)[..., t, :]
                    if ((cnewt > (1-tau) * cold[..., t, :]).any() or (snewt < (1-tau) * sold[..., t, :]).any()): 
                        # check if the inequality holds, with some thresholds
                        failed = True
                        break
                    xnew[..., t, :] = xnewt = self.f_fn(xnewt, unewt)[0]
                        
            if (failed):
                continue
            else:
                if (self.infeas):
                    for i in range(self.N): 
                        logcost -= alg.mu * ynew[i].log().sum()
                        cnew[i] = fp.computec(xnew[i], unew[i])
                        err += torch.linalg.vector_norm(cnew[i]+ynew[i], 1)
                    err = torch.maximum(alg.tol, err)
                else:
                    logcost = - self.mu * cnew.log().sum(-1).sum(-1)

                # step filter
                candidate = torch.stack((logcost, err), dim=-1)
                if torch.any( torch.all(candidate-torch.tile(torch.Tensor([1e-13, 0.]), B + (1,))>=self.filter, -1) ):
                    # relax a bit for numerical stability, strange
                    # todo: any for each sample in a batch?
                    failed=True
                    continue                    
                else:
                    idx=torch.all(candidate<=self.filter,-1)
                    self.filter = self.filter[~idx]
                    if self.filter.ndim <= 2:  # todo: change this walkaround
                        self.filter = self.filter.unsqueeze(0)
                    self.filter=torch.cat((self.filter, candidate.unsqueeze(-2)), dim=-2)
                    break
                  
        if (failed):
            self.stepsize, self.failed= 0.0, failed
        else:
            self.cost = self.q_fn(self.x[...,:-1,:], self.u).sum(-1) + self.p_fn(self.x[...,-1,:],torch.zeros_like(self.u[...,-1,:])).sum(-1)
            self.x, self.u, self.y, self.s, self.c = xnew, unew, ynew, snew, cnew 
            self.err, self.stepsize, self.step, self.failed = err, stepsize, step, False


    def optimizer(self):
        r'''
        Call forwardpass and backwardpass to solve trajectory
        '''
        time_start = time.time()

        for t in range(self.T):
            self.x[...,t+1,:], _ = self.f_fn(self.x[...,t,:],self.u[...,t,:])
        self.c = self.c_fn(self.x[...,:-1,:], self.u)
        self.cost = self.q_fn(self.x[...,:-1,:], self.u).sum(-1) \
                        + self.p_fn(self.x[...,-1,:],torch.zeros_like(self.u[...,-1,:])).sum(-1)
        self.mu = self.cost/self.T/self.s[...,0,:].shape[-1]
        self.resetfilter(self.infeas, self.mu)
        self.reg, self.bp_failed, self.recovery = 0.0, False, 0

        for iter in range(self.maxiter):
            while True: 
                self.backwardpasscompact()
                if ~self.bp_failed: 
                    break    
                
            self.forwardpasscompact()
            time_used = time.time() - time_start
            # if (iter % 10 == 1):
            #     print('\n')
            #     print('Iteration','Time','mu','Cost','Opt. error','Reg. power','Stepsize')
            #     print('\n')
            #     print('%-12d%-12.4g%-12.4g%-12.4g%-12.4g%-12d%-12.3f\n'%(
            #             iter, time_used, self.alg.mu, self.fp.cost, self.bp.opterr, self.bp.reg, self.fp.stepsize))

            #-----------termination conditions---------------
            if (max(self.opterr, self.mu)<=self.tol):
                print("~~~Optimality reached~~~")
                break
            
            if (self.opterr <= 0.2*self.mu):
                self.mu = max(self.tol/10.0, min(0.2*self.mu, pow(self.mu, 1.2) ) )
                self.resetfilter(self.infeas, self.mu)
                self.reg, self.bp_failed, self.recovery = 0.0, False, 0

            if iter == self.maxiter - 1:
                print("max iter", self.maxiter, "reached, not the optimal one!")

        return self.x

class ddpGrad:

    def __init__(self, sys, cons):
        self.system = sys
        self.constraint_flag = True
        self.constraint = cons
        self.contraction_flag = True

    def forward(self, fp_list, x_init):
        with torch.autograd.set_detect_anomaly(True): # for debug
            # self.fp = fp_best #todo: uncomment
            # self.bp = bp_best
            # self.alg = alg_best
            # self.fp.initialroll()
            # x_init = self.fp.x[0]
            self.prepare(fp_list)
            Ku, ku = self.ipddp_backward(mu=1e-3)
            x, u, cost, cons = self.ipddp_forward(x_init, Ku, ku)
        return x, u, cost, cons

    def prepare(self, fp_list):
        n_batch = len(fp_list)
        # fp = self.fp  #todo: uncomment
        # fp.computeall()
        self.c, self.s = torch.stack([fp_list[batch_id].c for batch_id in range(n_batch)],dim=0).squeeze(-1), \
                         torch.stack([fp_list[batch_id].s for batch_id in range(n_batch)],dim=0).squeeze(-1)
        # self.c = torch.randn(2, 5, 6)
        # self.s = 0.01 * torch.ones_like(self.c) 
        with torch.no_grad(): # detach
            self.Qxx_terminal = torch.stack([fp_list[batch_id].pxx for batch_id in range(n_batch)],dim=0)
            self.Qx_terminal = torch.stack([fp_list[batch_id].px.squeeze(-1) for batch_id in range(n_batch)],dim=0)
            # for t in range(self.T-1, -1, -1): 
            self.Q = torch.stack([
                                        torch.cat([torch.cat([fp_list[batch_id].qxx, fp_list[batch_id].qxu],dim=-1),
                                        torch.cat([fp_list[batch_id].qxu.mT, fp_list[batch_id].quu],dim=-1)], dim=-2) 
                                            for batch_id in range(n_batch)], dim=0) 
                # todo: vstack fp.qxx, fp.qxu, fp.quu 
            self.p  = torch.stack([
                                        torch.cat([fp_list[batch_id].qx, fp_list[batch_id].qu],dim=-2) 
                                            for batch_id in range(n_batch)], dim=0).squeeze(-1)
            # todo: vstack fp.qx, fp.qu  
            self.W = torch.stack([
                                        torch.cat([fp_list[batch_id].cx, fp_list[batch_id].cu],dim=-1) 
                                            for batch_id in range(n_batch)], dim=0) 
            # todo: vstack fp.cx, fp.cu
            # fx,fu,fxx,fxu,fuu = fp.fx, fp.fu, fp.fxx, fp.fxu, fp.fuu
            self.F = torch.stack([
                                        torch.cat([fp_list[batch_id].fx, fp_list[batch_id].fu],dim=-1) 
                                            for batch_id in range(n_batch)], dim=0) 
             # todo concatenate A,B
            self.G = torch.stack([
                                        torch.cat([torch.cat([fp_list[batch_id].fxx, fp_list[batch_id].fxu],dim=-1),
                                        torch.cat([fp_list[batch_id].fxu.mT, fp_list[batch_id].fuu],dim=-1)], dim=-2) 
                                            for batch_id in range(n_batch)], dim=0) 
            # todo second order dynamics
            self.T = self.F.size(-3)

    def ipddp_backward(self, mu):
        # Q: (B*, T, N, N), p: (B*, T, N), where B* can be any batch dimensions, e.g., (2, 3)
        B = self.p.shape[:-2]
        ns, nc = self.Qx_terminal.size(-1), self.F.size(-1) - self.Qx_terminal.size(-1)
        Ku = torch.zeros(B + (self.T, nc, ns), dtype=self.p.dtype, device=self.p.device)
        ku = torch.zeros(B + (self.T, nc), dtype=self.p.dtype, device=self.p.device)

        V, v = self.Qxx_terminal, self.Qx_terminal
        for t in range(self.T-1, -1, -1): 
            Ft = self.F[...,t,:,:]
            Qt = self.Q[...,t,:,:] + Ft.mT @ V @ Ft
            qt = self.p[...,t,:] + bmv(Ft.mT, v) 
            # if self.contraction_flag:
            #     Qt += torch.tensordot(v.mT, self.G[...,t,:,:,:], dims=-1) # todo :check!!!!
            if self.constraint_flag:
                Wt = self.W[...,t,:,:]
                st, ct = self.s[...,t,:], self.c[...,t,:] 
                r = st * ct + mu
                cinv = 1. / ct
                SCinv = torch.diag_embed(st * cinv)
                Qt += - Wt.mT @ SCinv @ Wt
                qt += bmv(Wt.mT, st) - bmv(Wt.mT, cinv * r)
            # if self.system.c1 is not None: # tocheck
            #     qt = qt + bmv(Ft.mT @ V, self.system.c1)

            Qxx, Qxu = Qt[..., :ns, :ns], Qt[..., :ns, ns:]
            Qux, Quu = Qt[..., ns:, :ns], Qt[..., ns:, ns:]
            qx, qu = qt[..., :ns], qt[..., ns:]

            Quu_inv = torch.linalg.pinv(Quu)
            Ku[...,t,:,:] = Kut = - Quu_inv @ Qux
            ku[...,t,:] = kut = - bmv(Quu_inv, qu)

            V = Qxx + Qxu @ Kut + Kut.mT @ Qux + Kut.mT @ Quu @ Kut
            v = qx  + bmv(Qxu, kut) + bmv(Kut.mT, qu) + bmv(Kut.mT @ Quu, kut)
            
        return Ku, ku

    def ipddp_forward(self, x_init, Ku, ku):
        assert x_init.device == Ku.device == ku.device
        assert x_init.dtype == Ku.dtype == ku.dtype
        assert x_init.ndim == 2, "Shape not compatible."
        B = self.p.shape[:-2]
        ns, nc = self.Qx_terminal.size(-1), self.F.size(-1) - self.Qx_terminal.size(-1)
        u = torch.zeros(B + (self.T, nc), dtype=self.p.dtype, device=self.p.device)
        cost = torch.zeros(B, dtype=self.p.dtype, device=self.p.device)
        x = torch.zeros(B + (self.T+1, ns), dtype=self.p.dtype, device=self.p.device)
        x[..., 0, :] = x_init
        xt = x_init

        self.system.set_refpoint(t=torch.Tensor([0.]))
        for t in range(self.T):
            Kut, kut = Ku[...,t,:,:], ku[...,t,:]
            u[..., t, :] = ut = bmv(Kut, xt) + kut 
            xut = torch.cat((xt, ut), dim=-1)
            x[...,t+1,:] = xt = self.system(xt, ut)[0]
            cost = cost + 0.5 * bvmv(xut, self.Q[...,t,:,:], xut) + (xut * self.p[...,t,:]).sum(-1)
        
        if self.constraint_flag:
            ncons = self.W.size(-2)
            cons = torch.zeros(B + (self.T, ncons), dtype=self.p.dtype, device=self.p.device )
            cons = self.constraint(x[...,0:-1,:], u)
            return x[...,0:-1,:], u, cost, cons
        else: 
            return x[...,0:-1,:], u, cost
        
