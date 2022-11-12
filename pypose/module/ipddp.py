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
        self.tol = tol
        self.infeas = infeas

class fwdPass:
    def __init__(self,sys=None, cost=None, cons=None):
        self.f = sys
        self.p = cost['stagecost']
        self.q = cost['terminalcost']
        self.c = cons

    def setPmat(self):
        return True
    
    def computenextx(self, x, u): # seems to be embedded in system
        return self.f(x, u)

    def computec(self):
        return 

    def computep(self):
        return True

    def computeq(self):
        return True
    
    def computeall(self):
        self.computeprelated()
        self.computefrelated()
        self.computeqrelated()
        self.computecrelated()

    def computeprelated(self):
        return True

    def computefrelated(self):
        return True

    def computeqrelated(self):
        return True

    def computecrelated(self):
        return True

    def initialroll(self):
        q = VectorXd::Zero(N)
        for i in range(N):
            x_temp = x[i]
            u_temp = u[i]
            c[i] = computecminvo(x_temp, u_temp, i, corridor)
            q(i) = computeq(x_temp, u_temp)  #  compute cost then used in resetfilter
            x[i+1] = computenextx(x_temp, u_temp)
        }
        self.cost = q.sum() + self.computep(x[N], x_d)
        self.costq = q.sum()

    def resetfilter(self, alg):
        self.logcost = self.cost
        self.err = 0.0
        if (alg.infeas):
            for i in range(N): 
                self.logcost -= alg.mu * self.y[i].array().log().sum()
                self.err += (self.c[i]+self.y[i]).lpNorm<1>()
            if (self.err < alg.tol):
                self.err = 0.0

        else:
            for i in range(N):
                self.logcost -= alg.mu * (-self.c[i]).array().log().sum()
                self.err = 0.0

        self.filter = [self.logcost, self.err]
        self.step = 0
        self.failed = False

    def finalroll(self):
        kerkCost = VectorXd::Zero(N);
        for i in range(self.N):
            x_temp = x[i]
            u_temp = u[i]
            # time2barR((u_temp.tail(1))(0))
            # jerkCost(i) = (u_temp.head(sys_order*dim).transpose() * barR * u_temp.head(sys_order*dim))(0); 

    # get function should be implemented by @property, should take care of deepcopy thing      

    def removeColumn(self, matrix, colToRemove):
        numRows = matrix.rows()
        numCols = matrix.cols()-1
        if( colToRemove < numCols ):
            matrix.block(0,colToRemove,numRows,numCols-colToRemove) = matrix.block(0,colToRemove+1,numRows,numCols-colToRemove)
        matrix.conservativeResize(numRows,numCols); 
        return matrix    


class bwdPass:
    def __init__(self):
        return True

    def resetreg(self):
        self.reg = 0.0
        self.failed = False
        self.recovery = 0

    def initreg(self, regvalue=1.0):
        self.reg = regvalue
        self.failed = False
        self.recovery = 0



class ddpOptimizer:
    def __init__(self, sys=None, cost=None, cons=None):
        self.alg = algParam()
        self.fp = fwdPass(sys=sys, cost=cost, cons=cons)
        self.bp = bwdPass()

    def backwardpass(self):
        fp = self.fp
        bp = self.bp
        alg = self.alg

        n_state = fp.n_state
        n_ctrl = fp.n_ctrl
        N = fp.N
        dV = [0.0,0.0]
        c_err = 0.0
        mu_err = 0.0
        Qu_err = 0.0

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
        for i in range(N-1, -1, -1):
                Qx = qx[i] + cx[i].transpose() * s[i] + fx[i].transpose() * Vx
                Qu = qu[i] + cu[i].transpose() * s[i] + fu[i].transpose() * Vx # (5b)

                fxiVxx = fx[i].transpose() * Vxx
                Qxx = qxx[i] + fxiVxx * fx[i]
                Qxu = qxu[i] + fxiVxx * fu[i]
                Quu = quu[i] + fu[i].transpose() * Vxx * fu[i]  # (5c-5e)
                Quu = 0.5 * (Quu + Quu.transpose())

                # todo S = s[i].asDiagonal();
                Quu_reg = Quu + (pow(fp.reg_exp_base, bp.reg) - 1) * np.eye(n_ctrl, n_ctrl)

                if (alg.infeas):
                    r = np.multiply(s[i], y[i]) - alg.mu
                    rhat = np.multiply(s[i],  (c[i] + y[i]) ) - r
                    yinv = np.reciprocal( y[i] )
                    tempv1 = np.multiply( s[i], yinv)
                    SYinv = np.diag(tempv1)  # y is vector
                    cuitSYinvcui = cu[i].transpose() * SYinv * cu[i]
                    SYinvcxi = SYinv * cx[i]

                    try: 
                        lltofQuuReg = np.linalg.cholesky(Quu_reg + cuitSYinvcui) # compute the Cholesky decomposition 
                    except: 
                        bp.failed = True
                        bp.opterr = np.Inf  # todo, assign fp, bp 
                        return

                    tempv2 = np.multiply(yinv, rhat)
                    Qu += cu[i].transpose() * tempv2
                    tempQux = Qxu.transpose() + cu[i].transpose() * SYinvcxi
                    tempm = np.hstack( (Qu, tempQux) )

                    # kK = - lltofQuuReg.solve(tempm)
                    kK = - np.linalg.solve(Quu_reg + cuitSYinvcui, tempm)

                    ku = kK[:,0]
                    Ku = kK[:,1:]
                    cuiku = cu[i]*ku
                    cxiPluscuiKu = cx[i] + cu[i]*Ku

                    bp.ks[i] = np.multiply( yinv, (rhat + np.multiply( s[i], cuiku)) )

                    bp.Ks[i] = SYinv * cxiPluscuiKu
                    bp.ky[i] = - (c[i] + y[i]) - cuiku
                    bp.Ky[i] = -cxiPluscuiKu

                    Quu = Quu + cuitSYinvcui
                    Qxu = tempQux.transpose() # Qxu + cx[i].transpose() * SYinvcui
                    Qxx += cx[i].transpose() * SYinvcxi
                    Qx += cx[i].transpose() * tempv2

                else:
                    r = np.multiply(s[i],  c[i]) + alg.mu
                    cinv = np.reciprocal( c[i] )
                    tempv1 = np.multiply(s[i],  cinv)
                    SCinv = np.diag(tempv1) #  y is vector
                    SCinvcui = SCinv * cu[i]
                    SCinvcxi = SCinv * cx[i]
                    cuitSCinvcui = cu[i].transpose() * SCinvcui
                    
                    try:
                        lltofQuuReg = np.linalg.cholesky(Quu_reg - cuitSCinvcui) # compute the Cholesky decomposition 
                    except: 
                        bp.failed = True
                        bp.opterr = np.Inf
                        return

                    tempv2 = np.multiply(cinv, r)
                    Qu -= cu[i].transpose() * tempv2 # (12b)            
                    tempQux = Qxu.transpose() - cu[i].transpose() * SCinvcxi
                    temp = np.hstack(( Qu, tempQux))

                    # kK = - lltofQuuReg.solve(temp)
                    kK = - np.linalg.solve(lltofQuuReg, temp)

                    ku = kK[:,0]
                    Ku = kK[:,1:]
                    cuiku = cu[i]*ku
                    bp.ks[i] = -np.multiply(cinv, (r + np.multiply( s[i], cuiku)) )
                    bp.Ks[i] = - (SCinv * (cx[i] + cu[i] * Ku)) # (11) checked
                    bp.ky[i] = np.zeros(c[i].size(), 1)
                    bp.Ky[i] = np.zeros(c[i].size(), n_state)       
                    Quu = Quu - cuitSCinvcui # (12e)
                    Qxu = tempQux.transpose(); # Qxu - cx[i].transpose() * SCinvcui; // (12d)
                    Qxx -= cx[i].transpose() * SCinvcxi # (12c)
                    Qx -= cx[i].transpose() * tempv2 # (12a)
                

                dV[0] += (ku.transpose() * Qu)(0)

                QxuKu = Qxu * Ku
                KutQuu = Ku.transpose() * Quu

                dV[1] += (0.5 * ku.transpose() * Quu * ku)(0)
                Vx = Qx + Ku.transpose() * Qu + KutQuu * ku + Qxu * ku # (btw 11-12)
                Vxx = Qxx + QxuKu.transpose() + QxuKu + KutQuu * Ku # (btw 11-12)
                Vxx = 0.5 * ( Vxx + Vxx.transpose() ) # for symmetry

                bp.ku[i] = ku
                bp.Ku[i] = Ku

                Qu_err = np.maximum(Qu_err, np.linalg.norm(Qu, np.inf)  )
                mu_err = np.maximum(mu_err, np.linalg.norm(r, np.inf)  )
                if (alg.infeas):
                    c_err=np.maximum(c_err, np.linalg.norm(c[i]+y[i], np.inf) )

        bp.failed = False
        bp.opterr = np.maximum( np.maximum( Qu_err, c_err), mu_err)
        bp.dV = dV

        self.fp = fp
        self.bp = bp
        self.alg = alg
        return True

    def forwardpass(self):
        fp = self.fp
        bp = self.bp
        alg = self.alg

        N=fp.N

        xold, uold, yold, sold, cold=fp.x, fp.u, fp.y, fp.s, fp.c
        xnew, unew, ynew, snew, cnew=fp.x, fp.u, fp.y, fp.s, fp.c #todo: copy issue?

        cost, costq, logcost = 0., 0., 0.
        qnew = np.zeros(N,1)
        stepsize = 0.
        err = 0.
        tau = max(0.99, 1-alg.mu)
        steplist = pow(2.0, ArrayXd::LinSpaced(11, -10, 0).reverse() )
        failed = False

        for step in range(steplist.size()):
            failed = False
            stepsize = steplist(step)
            xnew[0] = xold[0]
            if (alg.infeas):
                for i in range(N):
                    ynew[i] = yold[i] + stepsize*bp.ky[i]+bp.Ky[i]*(xnew[i]-xold[i])
                    snew[i] = sold[i] + stepsize*bp.ks[i]+bp.Ks[i]*(xnew[i]-xold[i])

                    if (    (ynew[i].array()<(1-tau)*yold[i].array()).any() or 
                            (snew[i].array()<(1-tau)*sold[i].array()).any()   ): 
                        failed = True
                        break
                    
                    unew[i] = uold[i] + stepsize*bp.ku[i]+bp.Ku[i]*(xnew[i]-xold[i])
                    xnew[i+1] = fp.computenextx(xnew[i], unew[i])
            else:
                for i in range(N):
                    snew[i] = sold[i] + stepsize*bp.ks[i]+bp.Ks[i]*(xnew[i]-xold[i])
                    unew[i] = uold[i] + stepsize*bp.ku[i]+bp.Ku[i]*(xnew[i]-xold[i])
                    cnew[i] = fp.computec(xnew[i], unew[i], i)


                    if (    (cnew[i].array()>(1-tau)*cold[i].array()).any() or  
                            (snew[i].array()<(1-tau)*sold[i].array()).any()   ):
                        failed = True
                        break

                    xnew[i+1] = fp.computenextx(xnew[i], unew[i])
            
            
            if (failed):
                continue
            else:
                for i in range(N):
                    qnew[i] = fp.computeq(xnew[i], unew[i])
                cost = qnew.sum() + fp.computep(xnew[N], fp.x_d)
                costq = qnew.sum()
                logcost = cost  
                err = 0.0;            
                if (alg.infeas):
                    for i in range(N): 
                        logcost -= alg.mu * ynew[i].array().log().sum()
                        cnew[i] = fp.computec(xnew[i], unew[i], i)
                        err += (cnew[i]+ynew[i]).lpNorm<1>()
                    err = max(alg.tol, err)
                else:
                    for i in range(N):
                        cnew[i] = fp.computec(xnew[i], unew[i], i )
                        logcost -= alg.mu * (-cnew[i]).array().log().sum()
                    err=0.0
                

                candidate = [logcost, err]
                columnidtokeep = 0
                for i in range(fp.filter.cols()):
                    if (candidate(0)>=fp.filter(0, i) and candidate(1)>=fp.filter(1, i)):
                        failed=True
                        break;                    
                    else:
                        if (candidate(0)>fp.filter(0, i) or candidate(1)>fp.filter(1, i)):
                            columnidtokeep.push_back(i)
                    
                
                if (failed): continue

                tempm = zeros(2,columnidtokeep.size())
                for i in range(columnidtokeep.size() ): 
                    tempm.col(i) = fp.filter.col(columnidtokeep[i])
                fp.filter.resize(2, tempm.cols() + 1)
                fp.filter << tempm, candidate;            
                break
            
        
        if (failed):
            fp.failed=True
            fp.stepsize=0.0
        else:
            fp.cost, fp.costq, fp.logcost = cost, costq, logcost
            fp.x, fp.u, fp.y, fp.s, fp.c, fp.q = xnew, unew, ynew, snew, cnew, qnew 
            fp.err=err
            fp.stepsize=stepsize
            fp.step=step
            fp.failed=False
    
        return True

    def optimizer(self):
        iter = 0
        bp_no_upd_count = 0
        no_upd_count = 0
        bp_no_upd_count_max = 20
        opt_no_upd_count = 0
        opt_no_upd_count_max = 5

        for iter in range(self.alg.maxiter):
            while True: 
                self.backwardpass();
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

        return rtn      # todo: traj to be returned  



if __name__ == "__main__":
    N = 100    # Number of time steps

    # Time, Input, Initial state
    time  = torch.arange(0, N+1)
    input = torch.sin(2*math.pi*time/50)
    state = torch.tensor([1., 1.])

    # Create dynamics sys object
    sys = Floquet()
    # todo: cost class
    cost = 1.0
    solver = ddpOptimizer(sys, cost) 
    traj_opt = solver.optimizer()

    # Calculate trajectory
    state_all = torch.zeros(N+1, 2)
    state_all[0] = state
    obser_all = torch.zeros(N, 2)

    for i in range(N):
        state_all[i+1], obser_all[i] = sys(state_all[i], input[i])

    # Create time plots to show dynamics
    f, ax = plt.subplots(nrows=4, sharex=True)
    for _i in range(2):
        ax[_i].plot(time, state_all[:,_i], label='pp')
        ax[_i].set_ylabel(f'State {_i}')
    for _i in range(2):
        ax[_i+2].plot(time[:-1], obser_all[:,_i], label='pp')
        ax[_i+2].set_ylabel(f'Observation {_i}')
    ax[-1].set_xlabel('time')
    ax[-1].legend()

    # Jacobian computation - Find jacobians at the last step
    vars = ['A', 'B', 'C', 'D', 'c1', 'c2']
    sys.set_refpoint()
    [print(_v, getattr(sys, _v)) for _v in vars]

    # Jacobian computation - Find jacobians at the 5th step
    idx = 5
    sys.set_refpoint(state=state_all[idx], input=input[idx], t=time[idx])
    [print(_v, getattr(sys, _v)) for _v in vars]

    plt.show()