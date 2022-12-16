import sys
sys.path.append("..")
import torch as torch
import pypose as pp
import cvxpy as cp
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def LQR_cp(Q, p, F, f, x_init, T, n_state, n_ctrl): 

    tau = cp.Variable((n_state+n_ctrl, T))

    objs = []
    x0 = tau[:n_state,0]
    u0 = tau[n_state:,0]
    cons = [x0 == x_init]
    for t in range(T):
        xt = tau[:n_state,t]
        ut = tau[n_state:,t]
        objs.append(0.5*cp.quad_form(tau[:,t], Q[t]) + cp.sum(cp.multiply(p[t], tau[:,t])))
        if t+1 < T:
            xtp1 = tau[:n_state, t+1]
            cons.append(xtp1 == F[t]@tau[:,t]+f[t])
    prob = cp.Problem(cp.Minimize(sum(objs)), cons)
    prob.solve()
    assert 'optimal' in prob.status
    return torch.tensor(tau.value), torch.tensor([obj_t.value for obj_t in objs])


def test_LQR_linear():

    torch.manual_seed(0)
    n_batch, n_state, n_ctrl, T = 2, 4, 3, 5
    n_sc = n_state + n_ctrl
    Q = torch.randn(T, n_batch, n_sc, n_sc)
    Q = torch.matmul(Q.mT, Q)
    p = torch.randn(T, n_batch, n_sc)
    alpha = 0.2
    A = torch.tile(torch.eye(n_state) + alpha * torch.randn(n_state, n_state), (n_batch, 1, 1))
    B = torch.tile(torch.randn(n_state, n_ctrl), (n_batch, 1, 1))
    C = torch.tile(torch.eye(n_state), (n_batch, 1, 1))
    D = torch.tile(torch.zeros(n_state, n_ctrl), (n_batch, 1, 1))
    c1 = torch.tile(torch.randn(n_state), (n_batch, 1))
    c2 = torch.tile(torch.zeros(n_state), (n_batch, 1))
    x_init = torch.randn(n_batch, n_state)
    u_lqr = torch.tile(torch.zeros(n_ctrl), (n_batch, 1))
    F = torch.cat((A, B), axis=2)
    F = torch.tile(F, (T, 1, 1, 1))
    f = torch.tile(c1, (T, 1, 1))

    tau_cp, objs_cp = LQR_cp(Q[:,0], p[:,0], F[:,0], f[:,0], x_init[0], T, n_state, n_ctrl)
    tau_cp = tau_cp.T
    x_cp = tau_cp[:,:n_state]
    u_cp = tau_cp[:,n_state:]

    Q, p, A, B, C, D, c1, c2, x_init, u_lqr = [torch.Tensor(x).double() if x is not None else None
        for x in [Q, p, A, B, C, D, c1, c2, x_init, u_lqr]]

    lti = pp.module.LTI(A, B, C, D, c1, c2)
    
    LQR_DP  = pp.module.DP_LQR(n_state, n_ctrl, T, lti)
    x_lqr, u_lqr, objs_lqr, tau = LQR_DP(x_init, Q, p)
    tau_lqr = torch.cat((x_lqr, u_lqr), 2)
    print(x_lqr)
    assert torch.allclose(tau_cp, tau_lqr[:,0]) 

    print("Done")


def test_LQR_ltv_random():

    torch.manual_seed(1)
    n_batch, n_state, n_ctrl, T = 2, 4, 3, 5
    n_sc = n_state + n_ctrl

    Q = torch.randn(T, n_batch, n_sc, n_sc)
    Q = torch.matmul(Q.mT, Q)
    p = torch.randn(T, n_batch, n_sc)
    A = torch.randn(T, n_batch, n_state, n_state)
    B = torch.randn(T, n_batch, n_state, n_ctrl)
    C = torch.tile(torch.eye(n_state), (T, n_batch, 1, 1))
    D = torch.tile(torch.zeros(n_state, n_ctrl), (T, n_batch, 1, 1))
    c1 = torch.randn(T, n_batch, n_state)
    c2 = torch.tile(torch.zeros(n_state), (T, n_batch, 1))
    x_init = torch.randn(n_batch, n_state)
    
    class LTV(pp.module.System):
    
        def __init__(self, T, A, B, C, D, c1=None, c2=None):
            super(LTV, self).__init__()
            self.T = T
            self.A, self.B, self.C, self.D = A, B, C, D
            self.c1, self.c2 = c1, c2

        def forward(self, state, input):
            
            return super(LTV, self).forward(state, input)

        def state_transition(self, state, input, t=None):
            for t in range(self.T):
                A = self.A[t]
                B = self.B[t]
                c1 = self.c1[t]

            return torch.einsum('...ik,...k->...i', [A, state]) + torch.einsum('...ik,...k->...i', [B, input]) + c1

        def observation(self, state, input, t=None):
            for t in range(self.T):
                C = self.C[t]
                D = self.D[t]
                c2 = self.c2[t]

            return torch.einsum('...ik,...k->...i', [C, state]) + torch.einsum('...ik,...k->...i', [D, input]) + c2

        @property
        def A(self):
            return self._A

        @A.setter
        def A(self, A):
            self._A = A

        @property
        def B(self):
            return self._B

        @B.setter
        def B(self, B):
            self._B = B

        @property
        def C(self):
            return self._C

        @C.setter
        def C(self, C):
            self._C = C

        @property
        def D(self):
            return self._D

        @D.setter
        def D(self, D):
            self._D = D

        @property
        def c1(self):
            return self._c1

        @c1.setter
        def c1(self, c1):
            self._c1 = c1

        @property
        def c2(self):
            return self._c2

        @c2.setter
        def c2(self, c2):
            self._c2 = c2

    ltv = LTV(T, A, B, C, D, c1, c2)
    LQR_DP  = pp.module.DP_LQR(n_state, n_ctrl, T, ltv)
    x_lqr, u_lqr, objs_lqr, tau = LQR_DP(x_init, Q, p)

    print("Done")


if __name__ == '__main__':
    test_LQR_linear()
    test_LQR_ltv_random()