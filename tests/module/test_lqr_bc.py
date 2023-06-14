

import torch, pypose as pp

class TestLQR:

    def test_lqr_linear_bounded(self, device='cpu'):

        n_batch, T = 2, 5
        n_state, n_ctrl = 4, 3

        Q = torch.tile(torch.eye(n_state + n_ctrl, device=device), (n_batch, T, 1, 1))
        p = torch.tensor([
            [[-1.00, -0.68, -0.35, -1.42,  0.23, -1.73, -0.54],
             [-0.02, -1.13, -1.40,  0.13, -0.67, -0.92,  0.97],
             [-0.77,  0.59,  1.49,  0.54,  0.45,  0.62,  1.83],
             [-0.95, -0.80,  0.36, -0.81, -0.03,  0.39,  0.42],
             [ 1.52,  0.01,  0.36,  0.68,  0.20, -1.32, -0.60]],
            [[ 1.37,  0.38,  0.75,  1.55,  0.89,  0.82, -0.13],
             [ 0.68, -0.08,  0.81,  0.10, -0.05,  2.01, -0.64],
             [ 0.01,  0.68,  0.97, -1.15, -1.02, -0.87,  0.43],
             [-1.79, -1.08,  0.30,  0.32, -0.27,  1.36,  0.21],
             [ 0.20, -0.68, -2.73, -0.52, -0.27,  0.13,  0.86]]],
            device=device)
        A = torch.tensor([
            [[ 1.05, -0.19,  0.17,  0.31],
             [-0.01,  0.93,  0.04, -0.05],
             [ 0.02,  0.48,  1.26, -0.05],
             [ 0.07,  0.50, -0.53,  1.48]],
            [[ 1.54, -0.62,  0.78,  0.31],
             [-0.01,  0.01,  0.04, -0.77],
             [ 0.97,  0.48,  1.56, -0.28],
             [ 0.29,  0.88, -0.79,  1.48]]],
            device=device)
        B = torch.tensor([
            [[ 0.01, -0.99,  0.97], [-0.44, -0.10,  0.80], [-1.71,  2.33,  0.41], [-1.13, -0.93, -0.08]],
            [[ 0.04,  0.25,  0.12], [ 1.76,  1.42, -0.78], [-1.28, -0.21,  0.75], [-0.52,  0.64, -0.05]]],
            device=device)
        C = torch.tile(torch.eye(n_state, device=device), (n_batch, 1, 1))
        D = torch.zeros(n_batch, n_state, n_ctrl, device=device)
        c1 = torch.tensor([[ 0.25, -0.56, -0.95,  1.18], [ 0.76, -0.51, -0.95,  1.18]], device=device)
        c2 = torch.zeros(n_batch, n_state, device=device)
        x_init = torch.tensor([[ 1.50, -0.34, -2.18,  0.54], [-1.05, -1.36,  0.43,  0.80]], device=device)
        dt = 1
        u_lower = -torch.randn(n_batch, T, n_ctrl, device=device)
        u_upper = torch.randn(n_batch, T, n_ctrl, device=device)
        du = 0.1

        lti = pp.module.LTI(A, B, C, D, c1, c2).to(device)
        LQR = pp.module.LQR2(lti, Q, p, T).to(device)
        x, u, cost = LQR(x_init, dt, u_lower=u_lower, u_upper=u_upper, du=du)


if __name__ == '__main__':
    test = TestLQR()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test.test_lqr_linear_bounded(device)
