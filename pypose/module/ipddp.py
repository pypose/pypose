import time
import torch as torch
import torch.nn as nn
from ..basics import bmv, bvmv

class algParam:
    r'''
    The class of algorithm parameter.
    '''
    def __init__(self, mu=1.0, maxiter=45, tol=1.0e-7, infeas=False):
        self.mu = mu  
        self.maxiter = maxiter
        self.tol = torch.tensor(tol)
        self.infeas = infeas

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
        self.N = horizon
        self.n_state = n_state
        self.n_input = n_input
        self.n_cons = n_cons

        # initilize all the variables used in the forward pass
        # defined in dynamics function
        self.x = init_traj['state']
        self.u = init_traj['input']
        self.c = torch.zeros(self.N, self.n_cons, 1)
        self.y = 0.01*torch.ones(self.N, self.n_cons, 1)
        self.s = 0.1*torch.ones(self.N, self.n_cons, 1)
        self.mu = self.y*self.s

        # terms related with terminal cost
        self.p = torch.Tensor([0.0])
        self.px = torch.zeros(1, self.n_state)
        self.pxx = torch.eye(self.n_state, self.n_state)

        # terms related with system dynamics
        self.fx = torch.zeros(self.N, self.n_state, self.n_state)
        self.fu = torch.zeros(self.N, self.n_state, self.n_input)
        self.fxx = torch.zeros(self.N, self.n_state, self.n_state, self.n_state)
        self.fxu = torch.zeros(self.N, self.n_state, self.n_state, self.n_input)
        self.fuu = torch.zeros(self.N, self.n_state, self.n_input, self.n_input)

        # terms related with stage cost
        self.q = torch.zeros(self.N, 1)
        self.qx = torch.zeros(self.N, self.n_state, 1)
        self.qu = torch.zeros(self.N, self.n_input, 1)
        self.qxx = torch.zeros(self.N, self.n_state, self.n_state)
        self.qxu = torch.zeros(self.N, self.n_state, self.n_input)
        self.quu = torch.zeros(self.N, self.n_input, self.n_input)

        # terms related with constraint
        self.cx = torch.zeros(self.N, self.n_cons, self.n_state)
        self.cu = torch.zeros(self.N, self.n_cons, self.n_input)

        self.filter = torch.Tensor([[torch.inf], [0.]])
        self.err = 0.
        self.logcost = 0.
        self.step = 0
        self.failed = False
        self.stepsize = 1.0
        self.reg_exp_base = 1.6

    def computenextx(self, x, u):
        return self.f_fn(x, u)[0]

    def computec(self, x, u):
        return self.c_fn(x, u).mT

    def computep(self, x):
        return self.p_fn(x, torch.zeros(1, self.n_input)) # dummy input

    def computeq(self, x, u):
        return self.q_fn(x, u)
    
    def computeall(self):
        self.computeprelated()
        self.computefrelated()
        self.computeqrelated()
        self.computecrelated()

    def computeprelated(self): # terms related to the terminal cost
        self.p = self.computep(self.x[-1])
        self.p_fn.set_refpoint(state=self.x[-1], input=self.u[-1])
        self.px = self.p_fn.cx.mT
        self.pxx = self.p_fn.cxx.squeeze(0).squeeze(1)
        return 

    def computefrelated(self): # terms related with system dynamics
        for i in range(self.N):
            self.f_fn.set_refpoint(state=self.x[i], input=self.u[i])
            self.fx[i] = self.f_fn.A.squeeze(0).squeeze(1)
            self.fu[i] = self.f_fn.B.squeeze(0).squeeze(1)   
            self.fxx[i] = self.f_fn.fxx.squeeze(0).squeeze(1).squeeze(2)
            self.fxu[i] = self.f_fn.fxu.squeeze(0).squeeze(1).squeeze(2)
            self.fuu[i] = self.f_fn.fuu.squeeze(0).squeeze(1).squeeze(2)

    def computeqrelated(self): # terms related with stage cost
        for i in range(self.N):
            self.q[i] = self.q_fn(self.x[i], self.u[i])
            self.q_fn.set_refpoint(state=self.x[i], input=self.u[i])
            self.qx[i] = self.q_fn.cx.mT
            self.qu[i] = self.q_fn.cu.mT
            self.qxx[i] = self.q_fn.cxx # squeezed inside cxx definition
            self.qxu[i] = self.q_fn.cxu 
            self.quu[i] = self.q_fn.cuu

    def computecrelated(self): # terms related with constraints
        for i in range(self.N):
            self.c[i] = self.computec(self.x[i], self.u[i])
            self.c_fn.set_refpoint(state=self.x[i], input=self.u[i])
            self.cx[i] = self.c_fn.gx
            self.cu[i] = self.c_fn.gu        

    def initialroll(self):
        for i in range(self.N):
            x_temp = self.x[i]
            u_temp = self.u[i]
            self.c[i] = self.computec(x_temp, u_temp)
            self.q[i] = self.computeq(x_temp, u_temp)  #  compute cost then used in resetfilter
            self.x[i+1] = self.computenextx(x_temp, u_temp)
        self.cost = self.q.sum() + self.computep(self.x[-1])
        self.costq = self.q.sum()

    def resetfilter(self, alg):
        self.logcost = self.cost
        self.err = torch.Tensor([0.0])
        if (alg.infeas):
            for i in range(self.N): 
                self.logcost -= alg.mu * self.y[i].log().sum()
                self.err += torch.linalg.vector_norm(self.c[i]+self.y[i], 1)
            if (self.err < alg.tol):
                self.err = torch.Tensor([0.0])

        else:
            for i in range(self.N):
                self.logcost -= alg.mu * (-self.c[i]).log().sum()
                self.err = torch.Tensor([0.0])

        self.filter = torch.vstack((self.logcost, self.err))
        self.step = 0
        self.failed = False

class bwdPass:
    r'''
    The class used in backwardpass 
    here backwardpass means compute gains for next iteration from current trajectory
    (different from the backward in neural network, which computes the gradient) 
    '''
    def __init__(self, sys=None, cons=None, n_state=1, n_input=1, n_cons=0, horizon=1):
        self.f_fn = sys
        self.c_fn = cons
        self.N = horizon
        self.n_state = n_state
        self.n_input = n_input
        self.n_cons = n_cons

        self.reg = 0.0
        self.failed = False
        self.recovery = 0
        self.opterr = 0.
        self.dV = torch.zeros(1,2)

        self.ky = torch.zeros(self.N,  self.n_cons, 1)
        self.Ky = torch.zeros(self.N,  self.n_cons, self.n_state)
        self.ks = torch.zeros(self.N,  self.n_cons, 1)
        self.Ks = torch.zeros(self.N,  self.n_cons, self.n_state)
        self.ku = torch.zeros(self.N,  self.n_input,1)
        self.Ku = torch.zeros(self.N,  self.n_input,self.n_state)

    def resetreg(self):
        self.reg = 0.0
        self.failed = False
        self.recovery = 0

    def initreg(self, regvalue=1.0):
        self.reg = regvalue
        self.failed = False
        self.recovery = 0

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
        # self.system = system
        # self.constraint = constraint
        # self.T = T
        # self.Q = Q
        # self.p = p
        # self.constraint_flag = True
        # self.contraction_flag = True
        # self.W = torch.randn(2, 5, 6, 7) # todo:change

        self.alg = algParam()
        self.fp = fwdPass(sys=sys, stage_cost=stage_cost, terminal_cost=terminal_cost, cons=cons, n_state=n_state, n_input=n_input, n_cons=n_cons, horizon=horizon, init_traj=init_traj)
        self.bp = bwdPass(sys=sys,            cons=cons,n_state=n_state, n_input=n_input, n_cons=n_cons, horizon=horizon)
        self.N = horizon


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

    def optimizer(self):
        r'''
        Call forwardpass and backwardpass to solve trajectory
        '''
        time_start = time.time()
        self.fp.initialroll()
        self.alg.mu = self.fp.cost/self.fp.N/self.fp.s[0].shape[0]
        self.fp.resetfilter(self.alg)
        self.bp.resetreg()

        for iter in range(self.alg.maxiter):
            while True: 
                self.backwardpass()
                if ~self.bp.failed: 
                    break    
                
            self.forwardpass()
            time_used = time.time() - time_start
            # if (iter % 10 == 1):
            #     print('\n')
            #     print('Iteration','Time','mu','Cost','Opt. error','Reg. power','Stepsize')
            #     print('\n')
            #     print('%-12d%-12.4g%-12.4g%-12.4g%-12.4g%-12d%-12.3f\n'%(
            #             iter, time_used, self.alg.mu, self.fp.cost, self.bp.opterr, self.bp.reg, self.fp.stepsize))

            #-----------termination conditions---------------
            if (max(self.bp.opterr, self.alg.mu)<=self.alg.tol):
                print("~~~Optimality reached~~~")
                break
            
            if (self.bp.opterr <= 0.2*self.alg.mu):
                self.alg.mu = max(self.alg.tol/10.0, min(0.2*self.alg.mu, pow(self.alg.mu, 1.2) ) )
                self.fp.resetfilter(self.alg)
                self.bp.resetreg()

            if iter == self.alg.maxiter - 1:
                print("max iter reached, not the optimal one!")

        return self.fp, self.bp, self.alg

    def forward(self, x_init):
        # detached version to solve the best traj
        # with torch.no_grad(): #todo: uncomment
        #     fp_best,bp_best,alg_best = self.optimizer()

        with torch.autograd.set_detect_anomaly(True): # for debug
            # self.fp = fp_best #todo: uncomment
            # self.bp = bp_best
            # self.alg = alg_best
            # self.fp.initialroll()
            # x_init = self.fp.x[0]
            self.prepare()
            Ku, ku = self.ipddp_backward(mu=1e-3)
            x, u, cost, cons = self.ipddp_forward(x_init, Ku, ku)
        return x, u, cost, cons

    def prepare(self):
        fp = self.fp  #todo: uncomment
        fp.computeall()
        # self.c, self.s = fp.c, fp.s
        self.c = torch.randn(2, 5, 6)
        self.s = 0.01 * torch.ones_like(self.c) 
        with torch.no_grad(): # detach
            self.Qxx_terminal[..., :,:] = 0 # todo: vstack fp.pxx
            self.Qx_terminal[...,:,:] = 0   # todo: vstack fp.px
            for t in range(self.T-1, -1, -1): 
                self.Q[...,t,:,:] = 0 # todo: vstack fp.qxx, fp.qxu, fp.quu 
                self.p[...,t,:] = 0 # todo: vstack fp.qx, fp.qu  
                self.W[...,t,:,:] = 0 # todo: vstack fp.cx, fp.cu
                # fx,fu,fxx,fxu,fuu = fp.fx, fp.fu, fp.fxx, fp.fxu, fp.fuu
                self.F = 0 # todo concatenate A,B
                self.G = 0 # todo second order dynamics

    def ipddp_backward(self, mu):
        # Q: (B*, T, N, N), p: (B*, T, N), where B* can be any batch dimensions, e.g., (2, 3)
        B = self.p.shape[:-2]
        ns, nc = self.system.B.size(-2), self.system.B.size(-1)
        Ku = torch.zeros(B + (self.T, nc, ns), dtype=self.p.dtype, device=self.p.device)
        ku = torch.zeros(B + (self.T, nc), dtype=self.p.dtype, device=self.p.device)

        V, v = self.Qx_terminal, self.Qxx_terminal
        for t in range(self.T-1, -1, -1): 
            Ft = self.F[...,t,:,:]
            Qt = self.Q[...,t,:,:] + Ft.mT @ V @ Ft
            qt = self.p[...,t,:] + bmv(Ft.mT, v) 
            if self.contraction_flag:
                Qt += torch.tensordot(v.mT, self.G[...,t,:,:,:], dims=-1) # todo :check
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
        ns, nc = self.system.B.size(-2), self.system.B.size(-1)
        u = torch.zeros(B + (self.T, nc), dtype=self.p.dtype, device=self.p.device)
        cost = torch.zeros(B, dtype=self.p.dtype, device=self.p.device)
        x = torch.zeros(B + (self.T+1, ns), dtype=self.p.dtype, device=self.p.device)
        x[..., 0, :] = x_init
        xt = x_init

        self.system.set_refpoint(t=0)
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
        
