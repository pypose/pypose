import torch
import pypose.module.LQR as LQR
from torch.autograd.functional import jacobian as jacob
from torch import transpose as xpose
from torch import matmul as mult
from torch import squeeze as sq

import numpy as np
import matplotlib.pyplot as plt

class LQRClass(torch.nn.Module):
    def __init__(self, A, B, x0, C, c, T) -> None:
        super().__init__()
        self.A = torch.nn.Parameter(A.clone(), requires_grad = True)
        self.B = torch.nn.Parameter(B.clone(), requires_grad = True)
        self.x0 = x0
        self.zhat = None
        self.muhat = None
        self.z = None
        self.mu = None
        self.C = C
        self.c = c
        self.T = T
    
    def forward(self, A, B, x0):
        """ LQR forward, generates the trajectory and costates."""
        C = self.C # K
        c = self.c # k
        T = self.T

        n_state = B.size(0)
        n_ctrl = B.size(1)
        n_all = n_state + n_ctrl

        AB = torch.concat((A, B), dim = 1).unsqueeze(0).repeat(T - 1, 1, 1)
        AB = torch.block_diag(*AB)
        BL = torch.zeros(n_state * T, n_all * T)
        BL[:n_state,:n_state] = torch.eye(n_state)
        BL[n_state:n_state + AB.size(0), :AB.size(1)] = AB
        idx1c = torch.linspace(n_all, n_all * (T - 1), T - 1, dtype=int)
        idx1r = n_state * torch.linspace(1, T - 1, T - 1, dtype=int)
        for i in range(0,n_state):
            BL[idx1r + i, idx1c + i] = -1
        
        # Cuu = C[-1, -n_ctrl:, -n_ctrl:]
        # Cux = C[-1, -n_ctrl:, :n_state]
        # cu = c[-1, -n_ctrl:]
        # K = torch.linalg.solve(-Cuu, Cux)
        # k = torch.linalg.solve(-Cuu, cu)

        # BL[-n_ctrl:, -n_all:-n_all + n_state] = K
        # BL[-n_ctrl:, -n_ctrl:] = torch.eye(n_ctrl)

        # TL = torch.block_diag(*(C.unsqueeze(0).repeat(T, 1, 1)))
        TL = torch.block_diag(*C)

        Top = torch.concat((TL, torch.transpose(BL, 0, 1)), dim = 1)
        Bot = torch.concat((BL, torch.zeros(BL.size(0), BL.size(0))), dim = 1)
        FullMat = torch.concat((Top, Bot), dim = 0)

        b = torch.zeros(BL.size(0))
        b[:n_state] = x0
        # b[-n_ctrl:] = k
        # zers = torch.zeros(TL.size(0))
        solvec = torch.cat((-self.c.flatten(), b), dim = 0)

        # numpy.savetxt("SolVec.csv", solvec)

        # fullmat = numpy.genfromtxt('FullMat.csv', delimiter=' ')
        # solvec = numpy.genfromtxt('SolVec.csv', delimiter=' ')

        mu_z = torch.linalg.solve(FullMat, solvec)
        z, mu = mu_z[:TL.size(0)], mu_z[TL.size(0):]

        z2 = torch.reshape(z, (T, n_all))
        x, u = z2[:,:n_state], z2[:,-n_ctrl:]

        # return z, mu, BL
        return z, mu
        # z = [x1, u1, x2, u2, ..., xN, uN], mu = [mu1, mu2, ..., muN]

    def setTrueTraj(self, A, B, x0):
        self.zhat, self.muhat = self.forward(A, B, x0)
    
    def _setPredTraj(self, x0):
        self.z, self.mu = self.forward(self.A, self.B, x0)

    def _L(self, z):
        # return torch.sqrt(torch.sum((self.zhat - sq(z)) ** 2, dim=-1))/z.size()[0]
        return torch.sqrt(torch.sum((self.zhat - sq(z)) ** 2))

    def _J(self, z):
        # C = torch.block_diag(*(self.C.unsqueeze(0).repeat(self.T, 1, 1)))
        return sq(0.5 * mult(mult(z, torch.block_diag(*self.C)), z) \
               + torch.dot(z, self.c.flatten()))

    def _G(self, A, B, z):
        C = self.C[-1]
        c = self.c[-1]
        T = self.T

        n_state = B.size(0)
        n_ctrl = B.size(1)
        n_all = n_state + n_ctrl

        AB = torch.concat((A, B), dim = 1).unsqueeze(0).repeat(T - 1, 1, 1)
        AB = torch.block_diag(*AB)
        BL = torch.zeros(n_state * T, (n_all) * T)
        BL[:n_state,:n_state] = torch.eye(n_state)
        BL[n_state:n_state + AB.size(0), :AB.size(1)] = AB
        idx1c = torch.linspace(n_all, n_all * (T - 1), T - 1, dtype=int)
        idx1r = n_state * torch.linspace(1, T - 1, T - 1, dtype=int)
        for i in range(0,n_state):
            BL[idx1r + i, idx1c + i] = -1
        
        # Cuu = C[-n_ctrl:, -n_ctrl:]
        # Cux = C[-n_ctrl:, :n_state]
        # cu = c[-n_ctrl:]
        # K = torch.linalg.solve(-Cuu, Cux)
        # k = torch.linalg.solve(-Cuu, cu)

        # BL[-n_ctrl:, -n_all:-n_all + n_state] = -K
        # BL[-n_ctrl:, -n_ctrl:] = torch.eye(n_ctrl)

        # TL = torch.block_diag(*(C.unsqueeze(0).repeat(T, 1, 1)))

        # Top = torch.concat((TL, torch.transpose(BL, 0, 1)), dim = 1)
        # Bot = torch.concat((BL, torch.zeros(BL.size(0), BL.size(0))), dim = 1)
        # FullMat = torch.concat((Top, Bot), dim = 0)

        # x0 = z[:n_state]
        b = torch.zeros(BL.size(0))
        b[:n_state] = self.x0
        # b[-n_ctrl:] = k
        # zers = torch.zeros(TL.size(0))
        # solvec = torch.cat((zers, b), dim = 0)

        # numpy.savetxt("SolVec.csv", solvec)

        # fullmat = numpy.genfromtxt('FullMat.csv', delimiter=' ')
        # solvec = numpy.genfromtxt('SolVec.csv', delimiter=' ')

        # mu_z = torch.linalg.solve(torch.tensor(FullMat), torch.tensor(solvec))
        # z = mu_z[:TL.size(0)]

        # return mult(BL, z) - b
        return mult(BL, z) - b

    ### Derivates of loss and constraint functions
    def _H(self, A, B, z):
        return self._Jz(z) + sq(mult(xpose(self.mu,0,1), sq(self._Gz(A, B, z))))

    def _Lz(self, z):
        return sq(jacob(self._L, z, create_graph=True, vectorize=True))

    def _Gz(self, A, B, z):
        return sq(jacob(self._G, (A, B, z), create_graph=True, vectorize=True)[2])
    
    def _Gzz(self, A, B, z):
        return sq(jacob(self._Gz, (A, B, z), create_graph=True, vectorize=True)[2])
    
    def _Gtheta(self, A, B, z):
        dA, dB, _ = jacob(self._G, (A, B, z), create_graph=True, vectorize=True)
        return sq(dA), sq(dB)
    
    def _Jz(self, z):
        return sq(jacob(self._J, z, create_graph=True, vectorize=True))
    
    def _Jzz(self, z):
        return sq(jacob(self._Jz, z, create_graph=True, vectorize=True))
    
    def _Htheta(self, A, B, z):
        dA, dB, _ = jacob(self._H, (A, B, z))
        return sq(dA), sq(dB)
    
    def _lams(self):
        '''
        coefficient matrix = 
                        dim1                   dim2
             |-----------------------|---------------------|
        dim3 | Jzz + mu^T * Gzz  |         Gz^T         |
             |-----------------------|---------------------|
        dim4 |           Gz           |          0          |
             |-----------------------|---------------------|

        dim2 = dim4
        dim3 = number of states
        dim4 = number of inputs
        '''

        # Calculate matrix elements
        self.mu = torch.unsqueeze(self.mu, -1)
        tl = sq(self._Jzz(self.z)) + sq(mult(xpose(self.mu, 0, 1), \
             torch.movedim(sq(self._Gzz(self.A, self.B, self.zhat)), 0, 1)))
        bl = sq(self._Gz(self.A, self.B, self.z))
        tr = xpose(bl,0,1)
        dim2 = bl.size(dim=0)
        br = torch.zeros(dim2, tr.size(dim=1))
        dim3 = self.z.size(dim=0)
        dim4 = self.mu.size(dim=0)
        
        # Concatenate rhs matrix
        l = torch.cat((tl, bl), dim=0)
        r = torch.cat((tr, br), dim=0)
        coef_mat = torch.cat((l, r), dim=1)
        # lhs vector
        sol_vec = torch.cat((self._Lz(self.z), torch.zeros(coef_mat.size(0) - self._Lz(self.z).size(0))))
        # solve and return solution
        lams = torch.linalg.solve(-coef_mat, sol_vec)
        lamZ, lamMu = torch.tensor(lams[:dim3]), torch.tensor(lams[dim3:])
        return lamZ[:,None], lamMu[:,None]

    def adjoint(self, x0):
        self._setPredTraj(x0)
        lamZ, lamMu = self._lams()
        dGdA, dGdB = self._Gtheta(self.A, self.B, self.z)
        dHdA, dHdB = self._Htheta(self.A, self.B, self.z)
        dLdA = sq(mult(xpose(lamZ,0,1).unsqueeze(0), torch.movedim(dHdA, 0, 1))) \
               + sq(mult(xpose(lamMu,0,1).unsqueeze(0), torch.movedim(dGdA, 0, 1)))
        dLdB = sq(mult(xpose(lamZ,0,1).unsqueeze(0), torch.movedim(dHdB, 0, 1))) \
               + sq(mult(xpose(lamMu,0,1).unsqueeze(0), torch.movedim(dGdB, 0, 1)))
        return dLdA.detach(), dLdB.detach()
    
    def cheatBackwards(self, x0):
        dLdA, dLdB = self.adjoint(x0)
        self.A.grad, self.B.grad = dLdA, dLdB

def main():
    torch.manual_seed(0)
    
    n_state = 3
    n_ctrl = 3
    n_all = n_state + n_ctrl
    alpha = 0.2
    T = 5

    C = torch.randn(T, 1, n_all, n_all)
    C = torch.matmul(C.mT, C)
    c = torch.randn(T, 1, n_all)

    # A = torch.eye(n_state) + alpha * torch.randn(n_state, n_state)
    B = torch.randn(n_state, n_ctrl)
    A = torch.randn(n_state, n_state)
    x0 = torch.randn(n_state)

    LQRLearner = LQRClass(A, B, x0, sq(C), sq(c), T)
    z_pred, mu_pred, FullMat = LQRLearner.forward(A, B, x0)

    x0 = x0.unsqueeze(0)
    F = torch.cat((A, B), dim=1) \
            .unsqueeze(0).unsqueeze(0).repeat(T, 1, 1, 1)
    _lqr  = LQR.DP_LQR(n_state, n_ctrl, T, None, None)
    Ks, ks = _lqr.DP_LQR_backward(C, c, F, None)
    x_true, u_true, objs_true, tau_true = _lqr.DP_LQR_forward(x0, C, c, F, None, Ks, ks)
    x_true, u_true = torch.squeeze(x_true), torch.squeeze(u_true)
    z_true = torch.reshape(torch.concat((x_true, u_true), dim = 1),z_pred.size())
    mu_true = _lqr.DP_LQR_costates(tau_true, C, c, F)

    z1 = z_pred - z_true

    print("done")

def main1():

    torch.manual_seed(0)
    
    n_state = 3
    n_ctrl = 3
    n_all = n_state + n_ctrl
    alpha = 0.2
    T = 5

    C = sq(torch.randn(T, 1, n_all, n_all))
    C = torch.matmul(C.mT, C)
    c = sq(torch.randn(T, 1, n_all))
    
    expert = dict(
        C = C,
        c = c,
        A = (torch.eye(n_state) + alpha*torch.randn(n_state, n_state)),
        B = torch.randn(n_state, n_ctrl)
    )

    A = torch.eye(n_state) + alpha*torch.randn(n_state, n_state)
    B = torch.randn(n_state, n_ctrl)
    x0 = torch.randn(n_state)

    numIter = [100, 1000, 5000]
    changeX0vec = [False, True]
    
    for changeX0 in changeX0vec:
        for iter in numIter:
            lossZ = np.empty(iter, dtype = float)
            lossA = np.empty(iter, dtype = float)
            lossB = np.empty(iter, dtype = float)
            

            lqrLearn = LQRClass(A, B, x0, expert['C'], expert['c'], T)
            lqrLearn.setTrueTraj(expert['A'], expert['B'], x0)
            optim = torch.optim.RMSprop(lqrLearn.parameters(), lr=1e-3)

            lqrLearn2 = LQRClass(A, B, x0, expert['C'], expert['c'], T)
            lqrLearn2.setTrueTraj(expert['A'], expert['B'], x0)
            optim2 = torch.optim.RMSprop(lqrLearn2.parameters(), lr=1e-3)

            for i in range(0, iter):
                if changeX0:
                    x0 = torch.randn(n_state)

                lqrLearn.setTrueTraj(expert['A'], expert['B'], x0)
                lqrLearn.cheatBackwards(x0)

                lqrLearn2.setTrueTraj(expert['A'], expert['B'], x0)
                lqrLearn2._setPredTraj(x0)
                loss = lqrLearn2._L(lqrLearn2.z)
                loss.backward()

                
                optim.step()
                optim.zero_grad(set_to_none=True)

                lossZ[i] = lqrLearn._L(lqrLearn.z).detach().numpy()
                lossA[i] = torch.norm(lqrLearn.A - expert['A']).detach().numpy()
                lossB[i] = torch.norm(lqrLearn.B - expert['B']).detach().numpy()

                optim2.step()
                optim2.zero_grad(set_to_none=True)

                if (i % int(iter/10) == 0):
                    print("Iteration {iterationNum}".format(iterationNum = i))
                    print(lqrLearn._L(lqrLearn.z))
                    print(torch.norm(lqrLearn.A - expert['A']))
                    print(torch.norm(lqrLearn.B - expert['B']))

                    print(lqrLearn2._L(lqrLearn2.z))
                    print(torch.norm(lqrLearn2.A - expert['A']))
                    print(torch.norm(lqrLearn2.B - expert['B']))

                    print("\n")
            
            plt.figure()
            plt.plot(np.linspace(1,iter,iter), lossZ)
            plt.plot(np.linspace(1,iter,iter), lossA)
            plt.plot(np.linspace(1,iter,iter), lossB)
            plt.legend(["Loss in z", "norm(Expert_A - Pred_A", "norm(Expert_B - Pred_B"])
            fname = "Figures/LossCurve_Iter_{numIter}_X0Diff_{x0diff}.png"\
                    .format(numIter=iter, x0diff=changeX0)
            plt.savefig(fname, format = "png")

            del lqrLearn
            del lqrLearn2

    return 0

main1()