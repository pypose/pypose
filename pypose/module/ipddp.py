import time
import torch as torch
import torch.nn as nn
from ..basics import bmv, bvmv, btdot

class ddpOptimizer(nn.Module):
    r'''
    The class of ipddp optimizer
    iterates between forwardpass and backwardpass to get a final trajectory 
    '''
    # def __init__(self, system, constraint, Q, p, T):
    def __init__(self, sys=None, stage_cost=None, terminal_cost=None, cons=None, n_cons=0, init_traj=None):
        r'''
        Initialize three key classes
        '''
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
        self.mu, self.maxiter, self.tol, self.infeas = 1.0, 50, torch.tensor([1.0e-7]), False

        # quantities in forward pass
        self.c = torch.zeros(B + (self.T, ncons))
        self.y = 0.01 * torch.ones(B + (self.T, ncons))
        self.s = 0.1 * torch.ones(B + (self.T, ncons))
        # self.mu = self.y * self.s 
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
        # terms related with terminal cost
        self.px = torch.zeros(B + (ns,))
        self.pxx = torch.zeros(B + (ns, ns))
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
            self.err = torch.linalg.vector_norm(self.c + self.y, -1).sum(-1)
            if (self.err < self.tol):
                self.err = torch.zeros(self.x.shape[:-2])
        else:
            self.logcost = self.cost - self.mu * (-self.c).log().sum(-1).sum(-1)
            self.err = torch.zeros(self.x.shape[:-2])

        self.filter = torch.stack((self.logcost, self.err), dim=-1).unsqueeze(-2) #todo:check
        self.step = 0
        self.failed = False

    def backwardpasscompact(self, lastIterFlag=False):
        r'''
        Compute controller gains for next iteration from current trajectory.
        '''
        B = self.x.shape[:-2]
        if lastIterFlag: 
            B = self.p.shape[:-2]
            ns, nc = self.Qx_terminal.size(-1), self.F.size(-1) - self.Qx_terminal.size(-1)
            self.Ku = torch.zeros(B + (self.T, nc, ns), dtype=self.p.dtype, device=self.p.device)
            self.ku = torch.zeros(B + (self.T, nc), dtype=self.p.dtype, device=self.p.device)

        ns = self.x.shape[-1]
        if not lastIterFlag:
            c_err, mu_err, qu_err = torch.zeros(B), torch.zeros(B), torch.zeros(B)

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
                self.computeall()
                                    
            # backward recursions, similar to iLQR backward recursion, but more variables involved
            V, v = self.pxx, self.px
        else:
            V, v = self.Qxx_terminal, self.Qx_terminal

        for t in range(self.T-1, -1, -1):
            Ft = self.F[...,t,:,:]
            Qt = self.Q[...,t,:,:] + Ft.mT @ V @ Ft
            qt = self.p[...,t,:] + bmv(Ft.mT, v) 
            if self.contraction_flag: 
                Qt += btdot(v, self.G[...,t,:,:,:]) # todo :check!!!!
            if self.constraint_flag:
                qt += bmv(self.W[...,t,:,:].mT, self.s[...,t,:])

            if (self.infeas): #  start from infeasible/feasible trajs.
                Wt = self.W[...,t,:,:]
                st, ct, yt = self.s[...,t,:], self.c[...,t,:], self.y[...,t,:]  
                r, rhat, yinv = st * yt - self.mu, st * (ct + yt) - r, 1. / yt
                SYinv = torch.diag_embed(st * yinv)

                Qt += Wt.mT @ SYinv @ Wt
                qt += bmv(Wt.mT, yinv * rhat)
                Qxx, Qxu = Qt[...,:ns,:ns], Qt[...,:ns,ns:]
                Qux, Quu = Qt[...,ns:,:ns], Qt[...,ns:,ns:]
                qx, qu = qt[...,:ns], qt[...,ns:]                
                Quu_reg = Quu + self.Q[...,t,ns:,ns:] * (pow(self.reg_exp_base, self.reg) - 1.)
                   
                try: 
                    lltofQuuReg = torch.linalg.cholesky(Quu_reg) # compute the Cholesky decomposition 
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
                    Quu_reg = Quu + self.Q[...,t,ns:,ns:] * (pow(self.reg_exp_base, self.reg) - 1.) #todo:check     
                    try: #todo check batch output?
                        lltofQuuReg = torch.linalg.cholesky(Quu_reg) # compute the Cholesky decomposition 
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
                    self.ky[...,t,:] = torch.zeros(ct.shape[-1]) # omitted
                    self.Ky[...,t,:,:] = torch.zeros(ct.shape[-1], ns)       

            V = Qxx + Qxu @ Kut + Kut.mT @ Qux + Kut.mT @ Quu @ Kut
            v = qx  + bmv(Qxu, kut) + bmv(Kut.mT, qu) + bmv(Kut.mT @ Quu, kut)

            if not lastIterFlag:
                qu_err = torch.maximum(qu_err, torch.linalg.vector_norm(qu, float('inf'), dim=-1)  )
                mu_err = torch.maximum(mu_err, torch.linalg.vector_norm(r,  float('inf'), dim=-1)  )
                if (self.infeas): 
                    c_err=torch.maximum(c_err, torch.linalg.vector_norm(ct+yt, float('inf'), dim=-1) )
        if not lastIterFlag:
            self.bp_failed, self.opterr = False, torch.maximum(torch.maximum(qu_err, c_err), mu_err)

    def forwardpasscompact(self, lastIterFlag=False):
        r'''
        Compute new trajectory from controller gains.
        '''
        xold, uold, yold, sold, cold = self.x, self.u, self.y, self.s, self.c
        if not lastIterFlag:
            tau, steplist = torch.maximum(1.-self.mu,torch.Tensor([0.99])), pow(2.0, torch.linspace(-10, 0, 11).flip(0))
            B = self.x.shape[:-2]
        else: 
            steplist = torch.ones((1)) # skip linesearch
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
                for t in range(self.T): # forward recuisions
                    Kut, kut = self.Ku[...,t,:,:], self.ku[...,t,:]
                    unew[...,t,:] = unewt = uold[...,t,:] + stepsize * kut + bmv(Kut, xnewt - xold[...,t,:])
                    if not lastIterFlag:
                        Kst, kst = self.Ks[...,t,:,:], self.ks[...,t,:]
                        snew[...,t,:] = snewt = sold[...,t,:] + stepsize * kst + bmv(Kst, xnewt - xold[...,t,:])
                        cnew[...,t,:] = cnewt = self.c_fn(xnew[...,:-1,:], unew)[...,t,:]
                        if ((cnewt > (1-tau) * cold[...,t,:]).any() or (snewt < (1-tau) * sold[...,t,:]).any()): 
                            # todo: check
                            # check if the inequality holds, with some thresholds
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
                        err = torch.zeros(B)
                    # step filter
                    candidate = torch.stack((logcost, err), dim=-1)
                    if torch.any( torch.all(candidate-torch.tile(torch.Tensor([1e-13, 0.]), B + (1,))>=self.filter, -1) ):
                        # relax a bit for numerical stability, strange
                        # todo: any for each sample in a batch?
                        failed=True
                        continue                    
                    else:
                        idx = torch.all(candidate<=self.filter,-1)
                        
                        ### fixed: 
                        # todo: bug here!!! 
                        # wrong: self.filter[torch.logical_not(idx)]
                        # wrong: self.filter = self.filter[~idx]
                        # one walkaround: self.filter[idx] = torch.inf 
                        self.filter = self.filter[torch.logical_not(idx)]
                        if self.filter.ndim <= 2:  # todo: change this walkaround
                            self.filter = self.filter.unsqueeze(0)
                        self.filter=torch.cat((self.filter, candidate.unsqueeze(-2)), dim=-2)
                        break
                  
        if (failed):
            self.stepsize, self.failed= 0.0, failed
        else:
            self.x, self.u, self.y, self.s, self.c = xnew, unew, ynew, snew, cnew 
            self.cost, self.err, self.stepsize, self.step, self.failed = cost, err, stepsize, step, False

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
            # if (iter % 5 == 1):
            #     print('\n')
            #     print('Iteration','Time','mu','Cost','Opt. error','Reg. power','Stepsize')
            #     print('\n')
            # print('%-12d%-12.4g%-12.4g%-12.4g%-12.4g%-12d%-12.3f\n'%(
            #             iter, time_used, self.mu, self.cost, self.opterr, self.reg, self.stepsize))

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

        return self

    def forward(self, fp_list):
        with torch.autograd.set_detect_anomaly(True): # for debug
            self.prepare(fp_list)
            self.infeas = False
            self.backwardpasscompact(lastIterFlag=True)
            self.forwardpasscompact(lastIterFlag=True)
        return self.x, self.u, self.cost

    def prepare(self, fp_list):
        n_batch = len(fp_list)
        self.c, self.s = torch.cat([fp_list[batch_id].c for batch_id in range(n_batch)],dim=0), \
                         torch.cat([fp_list[batch_id].s for batch_id in range(n_batch)],dim=0)
        with torch.no_grad(): # detach
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
