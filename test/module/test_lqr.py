import torch, pypose as pp


class TestLQR:

    def test_lqr_linear(self, device='cpu'):

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
        x_init = torch.randn(n_batch, n_state, device=device)

        print(A.shape, B.shape, C.shape, D.shape, c1.shape, c2.shape)

        lti = pp.module.LTI(A, B, C, D, c1, c2)
        LQR = pp.module.LQR(lti).to(device)
        x, u, cost = LQR(x_init, Q, p)
        
        assert torch.allclose(x_ref, x)
        assert torch.allclose(u_ref, u)

        print("Done")


    def test_lqr_ltv_random(self, device='cpu'):

        # The reference data
        x_ref = torch.tensor([
            [[-6.7540615797042847e-01, -5.0691682100296021e-01, 7.5898689031600952e-01, -1.2840508222579956e+00],
             [-5.8979380130767822e-01, 4.7968378663063049e-01, 4.9140143394470215e-01, 1.5389611721038818e+00],
             [7.0577168464660645e-01, 1.2108355760574341e-01, -4.1594886779785156e-01, -4.9780607223510742e-03],
             [-9.1280683875083923e-02, -3.1216004490852356e-01, -2.6045233011245728e-02, -2.4929654598236084e-01],
             [-8.2846939563751221e-01, -1.0467082262039185e-01, 4.7646835446357727e-03, 3.4996372461318970e-01]],
            [[-4.8931270837783813e-01, -1.4382526874542236e+00, -5.7033026218414307e-01, 8.0017691850662231e-01],
             [1.0735887289047241e+00, 7.5766277313232422e-01, 2.9580247402191162e-01, 2.6784765720367432e-01],
             [6.7208647727966309e-01, 6.3249444961547852e-01, 1.0311534404754639e+00, 8.7712675333023071e-01],
             [3.8980004191398621e-01, 5.2705928683280945e-02, 2.4217382073402405e-01, 1.0259031057357788e+00],
             [-7.5009906291961670e-01, -7.6115742325782776e-02, 2.1744665503501892e-01, -4.9836611747741699e-01]]],
            device=device)

        u_ref = torch.tensor([
            [[9.3025046586990356e-01, -8.3184355497360229e-01, -5.5526494979858398e-01],
            [-5.8084952831268311e-01, 1.7431840896606445e+00, 5.2921897172927856e-01],
             [-1.8309469521045685e-01, 6.0399073362350464e-01, -2.5268882513046265e-01],
             [2.6191076636314392e-01, 1.5980556607246399e-01, -2.5468763709068298e-01],
             [1.3268099725246429e-01, -2.7961313724517822e-01, -6.1580306291580200e-01]],
            [[-3.3134907484054565e-01, -7.0013701915740967e-01, -1.1543761491775513e+00],
             [5.0882947444915771e-01, 2.4764348566532135e-01, 5.4224240779876709e-01],
             [2.2596299648284912e-01, 7.4115723371505737e-02, 2.1080863475799561e-01],
             [6.8086898326873779e-01, 2.6450711488723755e-01, -6.3344031572341919e-02],
             [-1.4054971933364868e+00, 1.6244288682937622e+00, 1.1680284142494202e-01]]],
            device=device)

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

        assert torch.allclose(x_ref, x)
        assert torch.allclose(u_ref, u)

        print("Done")


if __name__ == '__main__':
    test = TestLQR()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test.test_lqr_linear(device)
    test.test_lqr_ltv_random(device)
