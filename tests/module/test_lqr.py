import torch, pypose as pp


class TestLQR:

    def test_lqr_linear(self, device='cpu'):

        # The reference data
        x_ref = torch.tensor([
            [[ 1.5000000000000000e+00, -3.4000000357627869e-01, -2.1800000667572021e+00,  5.4000002145767212e-01],
             [ 7.0821952819824219e-01, -3.8224798440933228e-01,  1.2187952995300293e+00, -2.8270375728607178e-01],
             [ 1.0639727115631104e+00, -1.2412147521972656e+00, -6.9346541166305542e-01, -1.5017517805099487e+00],
             [ 7.4383002519607544e-01, -1.3315110206604004e+00, -2.4556130170822144e-01, -4.0074110031127930e-02],
             [ 8.3880603313446045e-02, -2.1701562404632568e+00, -1.0819011926651001e-01, -1.8782937526702881e-01]],
            [[-1.0499999523162842e+00, -1.3600000143051147e+00,  4.3000000715255737e-01,  8.0000001192092896e-01],
             [ 6.8658864498138428e-01, -2.7504599094390869e+00, -1.4547507762908936e+00, -1.2115331888198853e+00],
             [ 2.0442225933074951e+00, -2.2286529541015625e+00, -2.0166194438934326e+00, -2.4150242805480957e+00],
             [ 3.1830101013183594e+00,  1.3153430223464966e+00, -2.0461211204528809e+00, -1.9447833299636841e+00],
             [ 2.4065117835998535e+00, -6.8622183799743652e-01,  4.3126922845840454e-01,  1.2989995479583740e+00]]],
            device=device)

        u_ref = torch.tensor([
            [[ 8.0248779058456421e-01,  2.4977316856384277e+00,  1.5325248241424561e+00],
             [ 1.0684574842453003e+00,  2.7619558572769165e-01,  1.4526504278182983e-01],
             [-1.1169517040252686e+00,  1.0167986154556274e-01, -1.6897654533386230e-01],
             [ 5.0239115953445435e-02,  8.5282796621322632e-01, -3.1150743365287781e-01],
             [-2.0000000298023224e-01,  1.3200000524520874e+00,  6.0000002384185791e-01]],
            [[ 1.4756069183349609e+00, -1.2672777175903320e+00,  3.1231970787048340e+00],
             [ 1.4230382442474365e-01, -8.6675214767456055e-01,  2.0238971710205078e+00],
             [ 4.8898696899414062e-02,  4.5528823137283325e-01,  8.2487571239471436e-01],
             [ 3.2995104789733887e-02, -1.0528842210769653e+00,  1.7456316947937012e-01],
             [ 2.7000001072883606e-01, -1.2999999523162842e-01, -8.6000001430511475e-01]]],
            device=device)

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

        lti = pp.module.LTI(A, B, C, D, c1, c2).to(device)

        current_x = torch.zeros(n_batch, T, n_state, device=device)
        current_u = torch.zeros(n_batch, T, n_ctrl, device=device)
        current_x[...,0,:] = x_init

        for i in range(T-1):
            current_x[...,i+1,:], _ = lti(current_x[...,i,:], current_u[...,i,:])

        time  = torch.arange(0, T, device=device)

        LQR = pp.module.LQR(lti, Q, p, T).to(device)
        x, u, cost = LQR(x_init, current_x, current_u, time)

        torch.testing.assert_close(x_ref, x)
        torch.testing.assert_close(u_ref, u)


    def test_lqr_ltv(self, device='cpu'):

        # The reference data
        x_ref = torch.tensor([
            [[ 1.5000000000000000e+00, -3.4000000357627869e-01, -2.1800000667572021e+00,  5.4000002145767212e-01],
             [ 2.3775162696838379e+00,  5.3751611709594727e-01, -1.3024839162826538e+00,  1.4175162315368652e+00],
             [ 3.9909069538116455e+00,  3.1090658903121948e-01, -3.3690934181213379e+00,  2.0709068775177002e+00],
             [ 1.0442220687866211e+01, -5.9778088331222534e-01, -1.1637781143188477e+01,  4.6822204589843750e+00],
             [ 3.9795291900634766e+01, -4.3647155761718750e+00, -4.8524715423583984e+01,  1.6755289077758789e+01]],
            [[-1.0499999523162842e+00, -1.3600000143051147e+00,  4.3000000715255737e-01,  8.0000001192092896e-01],
             [-9.9928379058837891e-03, -3.1999289989471436e-01,  1.4700071811676025e+00,  1.8400070667266846e+00],
             [-7.5955772399902344e-01, -1.3795578479766846e+00,  2.2004423141479492e+00,  2.9404420852661133e+00],
             [-3.8077919483184814e+00, -5.6677923202514648e+00,  5.0722084045410156e+00,  7.2922077178955078e+00],
             [-1.7204711914062500e+01, -2.4644714355468750e+01,  1.8315288543701172e+01,  2.7195285797119141e+01]]],
            device=device)

        u_ref = torch.tensor([
            [[-6.1749362945556641e-01, 1.3425052165985107e+00,  1.5250453352928162e-01],
             [-1.0373514890670776e+00, 9.2264413833618164e-01, -2.6735547184944153e-01],
             [-1.0800623893737793e+00, 8.7994861602783203e-01, -3.1005311012268066e-01],
             [-1.0744695663452148e+00, 8.8553714752197266e-01, -3.0446565151214600e-01],
             [-2.3000000417232513e-01, 1.7300000190734863e+00,  5.4000002145767212e-01]],
            [[-5.6333011388778687e-01, 1.3966690301895142e+00,  2.0666818320751190e-01],
             [-1.0332590341567993e+00, 9.2673635482788086e-01, -2.6326334476470947e-01],
             [-1.0799088478088379e+00, 8.8010215759277344e-01, -3.0989956855773926e-01],
             [-1.0744656324386597e+00, 8.8554108142852783e-01, -3.0446171760559082e-01],
             [-2.3000000417232513e-01, 1.7300000190734863e+00,  5.4000002145767212e-01]]],
            device=device)

        n_batch, T = 2, 5
        n_state, n_ctrl = 4, 3

        Q = torch.tile(torch.eye(n_state + n_ctrl, device=device), (n_batch, 1, 1))
        p = torch.tensor([[-1.00, -0.68, -0.35, -1.42, 0.23, -1.73, -0.54],
                          [-1.00, -0.68, -0.35, -1.42, 0.23, -1.73, -0.54]], device=device)
        rt = torch.arange(1, T+1).view(T, 1, 1)
        A = rt * torch.tile(torch.eye(n_state, device=device), (n_batch, T, 1, 1))
        B = rt * torch.ones(n_batch, T, n_state, n_ctrl, device=device)
        C = torch.tile(torch.eye(n_state, device=device), (n_batch, T, 1, 1))
        D = torch.zeros(n_batch, T, n_state, n_ctrl, device=device)
        x_init = torch.tensor([[ 1.50, -0.34, -2.18,  0.54],
                               [-1.05, -1.36,  0.43,  0.80]], device=device)

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

        ltv = MyLTV(A, B, C, D).to(device)

        current_x = torch.zeros(n_batch, T, n_state, device=device)
        current_u = torch.zeros(n_batch, T, n_ctrl, device=device)
        current_x[...,0,:] = x_init

        for i in range(T-1):
            current_x[...,i+1,:], _ = ltv(current_x[...,i,:], current_u[...,i,:])

        time  = torch.arange(0, T, device=device)

        LQR  = pp.module.LQR(ltv, Q, p, T).to(device)
        x, u, cost = LQR(x_init, current_x, current_u, time)

        torch.testing.assert_close(x_ref, x, atol=1e-5, rtol=1e-3)
        torch.testing.assert_close(u_ref, u, atol=1e-5, rtol=1e-3)


    def test_nlqr_cartpole(self, device='cpu'):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # The reference data
        x_ref = torch.tensor([
            [[ 0.0000000000000000e+00,  0.0000000000000000e+00, 3.1415927410125732e+00,  0.0000000000000000e+00],
             [ 0.0000000000000000e+00,  7.8299985034391284e-05, 3.1415927410125732e+00,  3.9145703340182081e-05],
             [ 7.8299984807017609e-07, -1.5641693607904017e-04, 3.1415932178497314e+00, -7.8217039117589593e-05],
             [-7.8116948998285807e-07, -3.8382178172469139e-04, 3.1415925025939941e+00, -1.9194713968317956e-04],
             [-4.6193872549338266e-06, -7.2401645593345165e-04, 3.1415905952453613e+00, -3.6203706986270845e-04]]],
            device=device)

        u_ref = torch.tensor([[[1.7618140578269958e-01],
                              [-5.2810668945312500e-01],
                              [-5.1161938905715942e-01],
                              [-7.6544916629791260e-01],
                              [-8.6499989032745361e-02]]],device=device)

        class CartPole(pp.module.NLS):
            def __init__(self, dt, length, cartmass, polemass, gravity):
                super().__init__()
                self.tau = dt
                self.length = length
                self.cartmass = cartmass
                self.polemass = polemass
                self.gravity = gravity
                self.polemassLength = self.polemass * self.length
                self.totalMass = self.cartmass + self.polemass

            def state_transition(self, state, input, t=None):
                x, xDot, theta, thetaDot = state.squeeze()
                force = input.squeeze()
                costheta = torch.cos(theta)
                sintheta = torch.sin(theta)

                temp = (
                    force + self.polemassLength * thetaDot**2 * sintheta
                ) / self.totalMass
                thetaAcc = (self.gravity * sintheta - costheta * temp) / (
                    self.length * (4.0 / 3.0 - self.polemass * costheta**2 / self.totalMass)
                )
                xAcc = temp - self.polemassLength * thetaAcc * costheta / self.totalMass

                _dstate = torch.stack((xDot, xAcc, thetaDot, thetaAcc))

                return (state.squeeze() + torch.mul(_dstate, self.tau)).unsqueeze(0)

            def observation(self, state, input, t=None):
                return state

        dt = 0.01
        len = 1.5
        m_cart = 20
        m_pole = 10
        g = 9.81
        T = 5
        n_batch = 1
        n_state, n_ctrl = 4, 1

        Q = torch.tile(torch.eye(n_state + n_ctrl, device=device), (n_batch, T, 1, 1))
        p = torch.tensor([
            [[-0.8156,  0.5950,  2.4234, -0.7989, -0.1750],
            [-0.3609, -1.4080, -0.8199, -0.8010,  0.5285],
            [ 0.4752,  1.2613,  1.1394, -0.7973,  0.5124],
            [-1.2204, -0.4849, -1.1381, -0.9851,  0.7658],
            [ 0.1382, -1.0695, -1.2191,  0.5620,  0.0865]]],
            device=device)
        time  = torch.arange(0, T, device=device) * dt
        current_x = torch.zeros(1, T, n_state, device=device)
        current_u = torch.sin(time).unsqueeze(1).unsqueeze(0)
        x_init = torch.tensor([[0, 0, torch.pi, 0]], device=device)
        current_x[...,0,:] = x_init

        cartPoleSolver = CartPole(dt, len, m_cart, m_pole, g).to(device)

        for i in range(T-1):
            current_x[...,i+1,:], _ = cartPoleSolver(current_x[...,i,:], current_u[...,i,:])

        LQR = pp.module.LQR(cartPoleSolver, Q, p, T).to(device)
        x, u, cost = LQR(x_init, current_x, current_u, time)

        torch.testing.assert_close(x_ref, x, atol=1e-5, rtol=1e-3)
        torch.testing.assert_close(u_ref, u, atol=1e-5, rtol=1e-3)


if __name__ == '__main__':
    test = TestLQR()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test.test_lqr_linear(device)
    test.test_lqr_ltv(device)
    test.test_nlqr_cartpole(device)
