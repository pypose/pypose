import torch as torch
import torch.nn as nn
import pypose as pp
# from torch.autograd.functional import jacobian

from pypose.module.dynamics import System
import math
import numpy as np
import matplotlib.pyplot as plt

class algParam:
    def __init__(self, mu=1.0, maxiter=100, tol=1.0e-7, infeas=True):
        self.mu = mu  
        self.maxiter = maxiter
        self.tol = torch.tensor(tol)
        self.infeas = infeas

class fwdPass:
    def __init__(self,sys=None, cost=None, cons=None, horizon=1):
        self.f_fn = sys
        self.p_fn = cost
        self.q_fn = cost
        self.c_fn = cons
        self.N = horizon
        self.n_state = self.f_fn.A.size(-1) # todo
        self.n_input = self.f_fn.B.size(-1)
        self.n_cons = self.c_fn.gx.size(0)
        # defined in dynamics function
        self.x = torch.zeros(self.N+1, 1, self.n_state)
        self.u = torch.zeros(self.N,   1, self.n_input)
        self.c = torch.zeros(self.N, self.n_cons, 1)
        self.y = 0.01*torch.ones(self.N, self.n_cons, 1) 
        self.s = 0.1*torch.zeros(self.N, self.n_cons, 1) 
        self.mu = self.y*self.s

        self.p = torch.Tensor([0.0])
        self.px = torch.zeros(1, self.n_state)
        self.pxx = torch.eye(self.n_state, self.n_state)

        # VecOfMatXd f;     // f
        self.fx = torch.zeros(self.N, self.n_state, self.n_state)
        self.fu = torch.zeros(self.N, self.n_state, self.n_input)

        self.fxx = torch.zeros(self.N, self.n_state, self.n_state, self.n_state)
        self.fxu = torch.zeros(self.N, self.n_state, self.n_state, self.n_input)
        self.fuu = torch.zeros(self.N, self.n_state, self.n_input, self.n_input)

        self.q = torch.zeros(self.N, 1)
        self.qx = torch.zeros(self.N, self.n_state, 1)
        self.qu = torch.zeros(self.N, self.n_input, 1)
        self.qxx = torch.zeros(self.N, self.n_state, self.n_state)
        self.qxu = torch.zeros(self.N, self.n_state, self.n_input)
        self.quu = torch.zeros(self.N, self.n_input, self.n_input)

        self.cx = torch.zeros(self.N, self.n_cons, self.n_state)
        self.cu = torch.zeros(self.N, self.n_cons, self.n_input)

        self.filter = torch.Tensor([[torch.inf], [0.]])
        # self.cost = 0.
        # self.costq = 0. # defined in resetfilter function
        self.err = 0.
        self.logcost = 0.
        self.step = 0
        self.failed = False
        self.stepsize = 1.0

        # Eigen::VectorXd jerkCost;

        self.reg_exp_base = 1.0

    def computenextx(self, x, u): # seems to be embedded in system
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

    def computeprelated(self):
        self.p = self.computep(self.x[-1])
        self.px = self.p_fn.cx.mT # use mT here s.t. cx property is consistent with dynamics implementation
        self.pxx = self.p_fn.cxx
        return 

    def computefrelated(self):
        for i in range(self.N):
            self.f_fn.set_refpoint(state=self.x[i], input=self.u[i])
            # print('checkpoint', self.fx.size())
            self.fx[i] = self.f_fn.A
            self.fu[i] = self.f_fn.B   
            self.fxx[i] = self.f_fn.fxx
            self.fxu[i] = self.f_fn.fxu  
            self.fuu[i] = self.f_fn.fuu  

    def computeqrelated(self):
        for i in range(self.N):
            self.q[i] = self.q_fn(self.x[i], self.u[i])
            self.q_fn.set_refpoint(state=self.x[i], input=self.u[i])
            self.qx[i] = self.q_fn.cx.mT
            self.qu[i] = self.q_fn.cu.mT
            self.qxx[i] = self.q_fn.cxx
            self.qxu[i] = self.q_fn.cxu
            self.quu[i] = self.q_fn.cuu

    def computecrelated(self):
        for i in range(self.N):
            self.c[i] = self.computec(self.x[i], self.u[i])
            self.c_fn.set_refpoint(state=self.x[i], input=self.u[i])
            self.cx[i] = self.c_fn.gx
            self.cu[i] = self.c_fn.gu        

    def initialroll(self):
        q = torch.zeros(1,self.N)
        for i in range(self.N):
            x_temp = self.x[i]
            u_temp = self.u[i]
            self.c[i] = self.computec(x_temp, u_temp)
            self.q[i] = self.computeq(x_temp, u_temp)  #  compute cost then used in resetfilter
            print('x, u temp', x_temp, u_temp)
            print(self.computenextx(x_temp, u_temp), self.x[i+1] )
            self.x[i+1] = self.computenextx(x_temp, u_temp)
        self.cost = q.sum() + self.computep(self.x[N])
        self.costq = q.sum()

    def resetfilter(self, alg):
        self.logcost = self.cost
        self.err = 0.0
        if (alg.infeas):
            for i in range(N): 
                self.logcost -= alg.mu * self.y[i].log().sum()
                self.err += torch.linalg.vector_norm(self.c[i]+self.y[i], 1)
            if (self.err < alg.tol):
                self.err = 0.0

        else:
            for i in range(N):
                self.logcost -= alg.mu * (-self.c[i]).log().sum()
                self.err = 0.0

        self.filter = torch.vstack((self.logcost, self.err))
        self.step = 0
        self.failed = False

    # def finalroll(self):
    #     jerkCost = torch.zeros(1,N)
    #     for i in range(self.N):
    #         x_temp = self.x[i]
    #         u_temp = self.u[i]
    #         # time2barR((u_temp.tail(1))(0))
    #         # jerkCost(i) = (u_temp.head(sys_order*dim).transpose() * barR * u_temp.head(sys_order*dim))(0); 

    # get function should be implemented by @property, should take care of deepcopy thing      

    # def removeColumn(self, matrix, colToRemove):
    #     numRows = matrix.rows()
    #     numCols = matrix.cols()-1
    #     if( colToRemove < numCols ):
    #         matrix.block(0,colToRemove,numRows,numCols-colToRemove) = matrix.block(0,colToRemove+1,numRows,numCols-colToRemove)
    #     matrix.conservativeResize(numRows,numCols); 
    #     return matrix    


class bwdPass:
    def __init__(self, sys=None, cons=None, horizon=1):
        self.f_fn = sys
        self.c_fn = cons
        self.N = horizon
        self.n_state = self.f_fn.A.size(-1) # todo
        self.n_input = self.f_fn.B.size(-1)
        self.n_cons = self.c_fn.gx.size(0)

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

class ddpOptimizer:
    def __init__(self, sys=None, cost=None, cons=None, horizon=None):
        self.alg = algParam()
        self.fp = fwdPass(sys=sys, cost=cost, cons=cons, horizon=horizon)
        self.bp = bwdPass(sys=sys,            cons=cons, horizon=horizon)
        self.N = horizon

    def backwardpass(self):
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

        if (fp.failed or bp.failed):
            bp.reg += 1.0
        else:
            if (fp.step == 0):
                bp.reg -= 1.0
            else:
                if (fp.step <= 3):
                    bp.reg = bp.reg
                else:
                    bp.reg += 1.0

        if (bp.reg < 0.0):
            bp.reg = 0.0
        else:
            if (bp.reg > 24.0):
                bp.reg = 24.0

        if ~fp.failed:
            fp.computeall()
        
        x, u, c, y, s, mu = fp.x, fp.u, fp.c, fp.y, fp.s, fp.mu 
        # double V = fp.getp();
        Vx, Vxx = fp.px, fp.pxx
        fx,fu,fxx,fxu,fuu = fp.fx, fp.fu, fp.fxx, fp.fxu, fp.fuu
        qx,qu,qxx,qxu,quu = fp.qx, fp.qu, fp.qxx, fp.qxu, fp.quu   
        cx, cu = fp.cx, fp.cu

        # todo: * to @ ?
        for i in range(self.N-1, -1, -1):
            print('checkpoint optimizer', i)
            Qx = qx[i] + cx[i].mT.matmul(s[i]) + fx[i].mT.matmul(Vx)
            Qu = qu[i] + cu[i].mT.matmul(s[i]) + fu[i].mT.matmul(Vx) # (5b)

            fxiVxx = fx[i].mT.matmul(Vxx)
            Qxx = qxx[i] + fxiVxx.matmul(fx[i])
            Qxu = qxu[i] + fxiVxx.matmul(fu[i])
            Quu = quu[i] + fu[i].mT.matmul(Vxx).matmul(fu[i])  # (5c-5e)
            Quu = 0.5 * (Quu + Quu.mT)

            # todo S = s[i].asDiagonal();
            Quu_reg = Quu + (pow(fp.reg_exp_base, bp.reg) - 1) * torch.eye(self.n_input, self.n_input)

            if (alg.infeas):
                r = s[i] * y[i] - alg.mu
                rhat = s[i] * (c[i] + y[i]) - r
                yinv = 1. / y[i]
                tempv1 = s[i] * yinv
                SYinv = torch.diag(tempv1.squeeze())  # y is vector
                cuitSYinvcui = cu[i].mT.matmul(SYinv).matmul(cu[i])
                SYinvcxi = SYinv.matmul(cx[i])

                try: 
                    lltofQuuReg = torch.linalg.cholesky(Quu_reg + cuitSYinvcui) # compute the Cholesky decomposition 
                except: 
                    bp.failed = True
                    bp.opterr = torch.inf  # todo, assign fp, bp 
                    self.fp = fp
                    self.bp = bp
                    self.alg = alg
                    return

                tempv2 = yinv * rhat
                Qu += cu[i].mT.matmul(tempv2)
                tempQux = Qxu.mT + cu[i].mT.matmul(SYinvcxi)
                tempm = torch.hstack( (Qu, tempQux) )

                # kK = - lltofQuuReg.solve(tempm)
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
                SCinv = torch.diag(tempv1.squeeze()) #  y is vector
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

                # kK = - lltofQuuReg.solve(temp)
                kK = - torch.linalg.solve(lltofQuuReg, temp)

                ku = torch.unsqueeze(kK[:,0],-1)
                Ku = kK[:,1:]
                cuiku = cu[i].matmul(ku)
                bp.ks[i] = - cinv * (r + s[i] * cuiku)
                bp.Ks[i] = - SCinv.matmul(cx[i] + cu[i].matmul(Ku)) # (11) checked
                bp.ky[i] = torch.zeros(c[i].size(), 1)
                bp.Ky[i] = torch.zeros(c[i].size(), n_state)       
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
        print('backward success')
        bp.opterr = torch.maximum( torch.maximum( Qu_err, c_err), mu_err)
        bp.dV = dV

        self.fp = fp
        self.bp = bp
        self.alg = alg

    def forwardpass(self):
        fp = self.fp
        bp = self.bp
        alg = self.alg

        xold, uold, yold, sold, cold=fp.x, fp.u, fp.y, fp.s, fp.c
        xnew, unew, ynew, snew, cnew=fp.x, fp.u, fp.y, fp.s, fp.c #todo: copy issue?

        cost, costq, logcost = 0., 0., 0.
        qnew = torch.zeros(self.N, 1)
        stepsize = 0.
        err = 0.
        tau = max(0.99, 1-alg.mu)
        steplist = pow(2.0, torch.linspace(-10, 0, 11).flip(0) )
        failed = False

        for step in range(steplist.shape[0]):
            failed = False
            stepsize = steplist[step]
            xnew[0] = xold[0]
            if (alg.infeas):
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
                for i in range(self.N):
                    snew[i] = sold[i] + stepsize*bp.ks[i]+bp.Ks[i].matmul((xnew[i]-xold[i]).mT)
                    unew[i] = uold[i] + stepsize*bp.ku[i]+bp.Ku[i].matmul((xnew[i]-xold[i]).mT)
                    cnew[i] = fp.computec(xnew[i], unew[i])


                    if (    (cnew[i]>(1-tau)*cold[i]).any() or  
                            (snew[i]<(1-tau)*sold[i]).any()   ):
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
                logcost = cost  
                err = 0.0;            
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
                    err=0.0
                

                candidate = torch.vstack((logcost, err))
                columnidtokeep = 0
                for i in range(len(fp.filter)):
                    if (candidate[0]>=fp.filter[0, i] and candidate[1]>=fp.filter[1,i]):
                        failed=True
                        break                    
                    else:
                        idx=torch.all(candidate<=fp.filter,0) #todo: check
                        fp.filter = fp.filter[:,~idx] #todo: check
                        fp.filter=torch.hstack((fp.filter,candidate))
                        # if (candidate[0]>fp.filter[0, i] or candidate[1]>fp.filter[1, i]):
                        #     columnidtokeep.push_back(i)
                    
                
                if (failed): continue

                tempm = torch.zeros(2,columnidtokeep.size())
                for i in range(columnidtokeep.size() ): 
                    tempm[:, i]= fp.filter[:,i]
                fp.filter.resize(2, tempm.shape + 1)
                fp.filter = torch.hstack((tempm, candidate))            
                break
            
        
        if (failed):
            fp.failed=True
            fp.stepsize=0.0
            print('forward failed')
        else:
            fp.cost, fp.costq, fp.logcost = cost, costq, logcost
            fp.x, fp.u, fp.y, fp.s, fp.c, fp.q = xnew, unew, ynew, snew, cnew, qnew 
            fp.err=err
            fp.stepsize=stepsize
            # fp.step=step
            fp.failed=False
            print('forward success')

        self.fp = fp
        self.bp = bp
        self.alg = alg

    def optimizer(self):
        self.fp.initialroll()
        self.alg.mu = self.fp.cost/self.fp.N/self.fp.s[0].shape[0]
        self.fp.resetfilter(self.alg)
        self.bp.resetreg()
        iter = 0
        bp_no_upd_count = 0
        no_upd_count = 0
        bp_no_upd_count_max = 20
        opt_no_upd_count = 0
        opt_no_upd_count_max = 5

        for iter in range(self.alg.maxiter):
            while True: 
                self.backwardpass()
                print('checkpoint bpreg', self.bp.reg)
                if ~self.bp.failed: 
                    break
                # in case dead loop in bp
                if (self.bp.reg == 24 and self.bp.failed):
                    bp_no_upd_count += 1
                else:
                    bp_no_upd_count = 0
                if (bp_no_upd_count > bp_no_upd_count_max):
                    break      

            self.forwardpass()

            #-----------termination conditions---------------
            if (max(self.bp.opterr, self.alg.mu)<=self.alg.tol):
                print("~~~Optimality reached~~~")
                break
            
            if (self.bp.opterr <= 0.2*self.alg.mu):
                self.alg.mu = max(self.alg.tol/10.0, min(0.2*self.alg.mu, pow(self.alg.mu, 1.2) ) )
                self.fp.resetfilter(self.alg)
                self.bp.resetreg()

            if (bp_no_upd_count > bp_no_upd_count_max):
                rtn = -4.0
                print("~~~ bp no update, terminate prematurely ~~~")
                break     

        return self.fp.x      # todo: traj to be returned  



if __name__ == "__main__":
    N = 5    # Number of time steps

    # Create dynamics sys object
    A = torch.eye(3, 3)
    B = torch.randn(3, 2)
    C = torch.randn(3, 3)
    D = torch.randn(3, 2)
    c1 = torch.zeros(1, 3)
    c2 = torch.randn(1, 3)
    state = torch.randn(1, 3)
    input = torch.randn(1, 2)
    lti = pp.module.LTI(A, B, C, D, c1, c2)    # Calculate trajectory
 
    n_state = state.size(1)
    n_input = input.size(1) 

    state_all = torch.zeros(N+1, n_state)
    input_all = torch.zeros(N, n_input)
    state_all[0] = state
    for i in range(N):
        # print('i', i, state_all[i])
        state_all[i+1], _ = lti(state_all[i], input_all[i])

    # Create cost object
    cx = torch.randn(1, 3)
    cu = torch.randn(1, 2)
    cxx = torch.eye(3, 3)
    cxx = cxx + cxx.mT
    cxu = torch.zeros(3, 2)
    cux = cxu.mT
    cuu = torch.eye(2, 2)
    cuu = cuu + cuu.mT
    c = torch.randn(1, 1)
    quadcost = pp.module.QuadCost(cx,cu,cxx,cxu,cux,cuu,c)

    # Create constraint object
    gx = torch.zeros( 2*n_input, n_state)
    gu = torch.vstack( (torch.eye(n_input, n_input), - torch.eye(n_input, n_input)) )
    g = torch.hstack( (1. * torch.ones(1, n_input), -1. * torch.ones(1, n_input)) )
    print('checkpoint', gx.size(), gu.size(), g.size())
    lincon = pp.module.LinCon(gx, gu, g)
    solver = ddpOptimizer(lti, quadcost, lincon, N) 

    traj_opt = solver.optimizer()



    # # Create time plots to show dynamics
    # f, ax = plt.subplots(nrows=4, sharex=True)
    # for _i in range(2):
    #     ax[_i].plot(time, state_all[:,_i], label='pp')
    #     ax[_i].set_ylabel(f'State {_i}')
    # for _i in range(2):
    #     ax[_i+2].plot(time[:-1], obser_all[:,_i], label='pp')
    #     ax[_i+2].set_ylabel(f'Observation {_i}')
    # ax[-1].set_xlabel('time')
    # ax[-1].legend()

    # # Jacobian computation - Find jacobians at the last step
    # vars = ['A', 'B', 'C', 'D', 'c1', 'c2']
    # sys.set_refpoint()
    # [print(_v, getattr(sys, _v)) for _v in vars]

    # # Jacobian computation - Find jacobians at the 5th step
    # idx = 5
    # sys.set_refpoint(state=state_all[idx], input=input[idx], t=time[idx])
    # [print(_v, getattr(sys, _v)) for _v in vars]

    # plt.show()