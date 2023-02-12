import torch, pypose as pp


class TestLQR:

    def test_lqr_linear(self, device='cpu'):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # The reference data
        x_ref = torch.tensor([
            [[-2.6332834362983704e-01, -3.4662923216819763e-01, 2.3803155422210693e+00, -4.2298097163438797e-02],
             [1.8492922186851501e-01, -1.3883942365646362e+00, 1.0898394584655762e+00, -1.6228721141815186e+00],
             [1.2138011455535889e+00, -7.1612375974655151e-01, 2.9535150527954102e-01, -6.8189740180969238e-01],
             [1.4839596748352051e+00, -1.1249105930328369e+00, -1.0301802158355713e+00, 9.8049354553222656e-01],
             [-3.4768491983413696e-01, -1.7062683105468750e+00, 4.6494283676147461e+00, 2.6779823303222656e+00]],
            [[-9.7435122728347778e-01, 4.9758589267730713e-01, 6.0299154371023178e-02, -5.2578479051589966e-01],
             [-6.3557159900665283e-01, 5.3919434547424316e-02, 7.2644424438476562e-01, -5.0483930110931396e-01],
             [-2.2745436429977417e-01, -1.6487598419189453e-01, 3.8718688488006592e-01, -4.6139001846313477e-01],
             [2.6967316865921021e-01, -3.5764670372009277e-01, 9.9902153015136719e-02, -4.5940721035003662e-01],
             [3.9155539870262146e-01, -2.0832152366638184e+00, 7.0109724998474121e-02, -5.4071712493896484e-01]]],
            device=device)

        u_ref = torch.tensor([
            [[1.0404593944549561e+00, 1.5860541164875031e-01, -1.2816929817199707e-01],
             [-1.4845380783081055e+00, -5.7454913854598999e-01, 2.5234240293502808e-01],
             [-6.3217121362686157e-01, -3.2808586955070496e-01, -3.6195179820060730e-01],
             [-1.6767632961273193e+00, 2.4053909778594971e+00, -1.0472261905670166e-01],
             [-1.7947965860366821e+00, 3.5269017219543457e+00, 9.0703134536743164e+00]],
             [[-1.7946422100067139e-01, 9.1525381803512573e-01, 1.7065811157226562e+00],
             [8.1361800432205200e-02, 4.0039587020874023e-01, 7.1144324541091919e-01],
             [4.3548654764890671e-02, 5.7816517353057861e-01, 1.0126723051071167e+00],
             [-3.0168235301971436e-01, -2.8970664739608765e-01, 7.2513574361801147e-01],
             [-7.2829566895961761e-02, 7.2901010513305664e-01, -3.1166392564773560e-01]]],
            device=device)

        torch.manual_seed(0)
        n_batch, T = 2, 5
        n_state, n_ctrl = 4, 3
        n_sc = n_state + n_ctrl

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
        x_init = torch.randn(n_batch, n_state, device=device)

        print(A.shape, B.shape, C.shape, D.shape, c1.shape, c2.shape)

        lti = pp.module.LTI(A, B, C, D, c1, c2)
        LQR = pp.module.LQR(lti).to(device)
        x, u, cost = LQR(x_init, Q, p)
        
        assert torch.allclose(x_ref, x, atol=1e-5)
        assert torch.allclose(u_ref, u, atol=1e-5)

        print("Done")


    def test_lqr_ltv_random(self, device='cpu'):

        torch.manual_seed(2)
        n_batch, n_state, n_ctrl, T = 2, 4, 3, 5
        n_sc = n_state + n_ctrl

        Q = torch.randn(n_batch, T, n_sc, n_sc, device=device)
        Q = Q.mT @ Q
        p = torch.randn(n_batch, T, n_sc, device=device)
        x_init = torch.randn(n_batch, n_state, device=device)

        A = torch.randn(n_batch, T, n_state, n_state, device=device)
        B = torch.randn(n_batch, T, n_state, n_ctrl, device=device)
        C = torch.tile(torch.eye(n_state, device=device), (n_batch, T, 1, 1))
        D = torch.tile(torch.zeros(n_state, n_ctrl, device=device), (n_batch, T, 1, 1))
        
        class MyLTV(pp.module.LTV):
        
            def __init__(self, A, B, C, D):
                super().__init__(A, B, C, D)

            @property
            def A(self):
                return self._A[...,self._t,:,:]

            @property
            def B(self):
                return self._B[...,self._t,:,:]

            @property
            def C(self):
                return self._C[...,self._t,:,:]
        
            @property
            def D(self):
                return self._D[...,self._t,:,:]

        ltv = MyLTV(A, B, C, D)
        lqr  = pp.module.LQR(ltv)
        x, u, cost = lqr(x_init, Q, p)

        print("Done")


if __name__ == '__main__':
    test = TestLQR()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test.test_lqr_linear(device)
    test.test_lqr_ltv_random(device)
