import torch, pypose as pp


class TestLQR:

    def test_lqr_linear(self, device):

        n_batch, T = 2, 5
        n_state, n_ctrl = 4, 3
        n_sc = n_state + n_ctrl
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        Q = torch.randn(n_batch, T, n_sc, n_sc, device=device)
        Q = torch.matmul(Q.mT, Q)
        p = torch.randn(n_batch, T, n_sc, device=device)
        A = torch.tile(torch.eye(n_state, device=device) \
            + 0.2 * torch.randn(n_state, n_state, device=device), (n_batch, 1, 1))
        B = torch.randn(n_batch, n_state, n_ctrl, device=device)
        C = torch.tile(torch.eye(n_state, device=device), (n_batch, 1, 1))
        D = torch.tile(torch.zeros(n_state, n_ctrl, device=device), (n_batch, 1, 1))
        c1 = torch.tile(torch.randn(n_state, device=device), (n_batch, 1))
        c2 = torch.tile(torch.zeros(n_state, device=device), (n_batch, 1))
        x0 = torch.randn(n_batch, n_state, device=device)

        print(A.shape, B.shape, C.shape, D.shape, c1.shape, c2.shape)

        lti = pp.module.LTI(A, B, C, D, c1, c2)
        LQR = pp.module.LQR(lti).to(device)
        x, u, cost = LQR(x0, Q, p)

        print("Done")


    def test_lqr_ltv_random(self, device):

        torch.manual_seed(1)
        n_batch, n_state, n_ctrl, T = 2, 4, 3, 5
        n_sc = n_state + n_ctrl

        Q = torch.randn(n_batch, T, n_sc, n_sc, device=device)
        Q = Q.mT @ Q
        p = torch.randn(n_batch, T, n_sc, device=device)
        x_init = torch.randn(n_batch, n_state, device=device)


        class MyLTV(pp.module.LTV):
        
            def __init__(self):
                super().__init__()

            @property
            def A(self):
                return torch.randn(n_batch, n_state, n_state, device=device)

            @property
            def B(self):
                return torch.randn(n_batch, n_state, n_ctrl, device=device)

            @property
            def C(self):
                return torch.tile(torch.eye(n_state, device=device), (n_batch, 1, 1))
        
            @property
            def D(self):
                return torch.tile(torch.zeros(n_state, n_ctrl, device=device), (n_batch, 1, 1))

            @property
            def c1(self):
                return torch.randn(n_batch, n_state, device=device)

            @property
            def c2(self):
                return torch.tile(torch.zeros(n_state, device=device), (n_batch, 1))

        ltv = MyLTV()
        lqr  = pp.module.LQR(ltv)
        x, u, cost = lqr(x_init, Q, p)

        print("Done")


if __name__ == '__main__':
    test = TestLQR()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test.test_lqr_linear(device)
    test.test_lqr_ltv_random(device)
