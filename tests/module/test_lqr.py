import torch, pypose as pp



class TestLQR:

    def test_lqr_linear(self, device='cpu'):

        # The reference data
        x_ref = torch.tensor([
            [[ 1.500000000e+00, -3.4000000357e-01, -2.1800000667e+00,  5.4000002145e-01],
             [ 7.082195281e-01, -3.8224798440e-01,  1.2187952995e+00, -2.8270375728e-01],
             [ 1.063972711e+00, -1.2412147521e+00, -6.9346541166e-01, -1.5017517805e+00],
             [ 7.438300251e-01, -1.3315110206e+00, -2.4556130170e-01, -4.0074110031e-02],
             [ 8.388060331e-02, -2.1701562404e+00, -1.0819011926e-01, -1.8782937526e-01],
             [-5.301788449e-02, -2.1380202770e+00, 1.54667758941+00,  -1.1694585084e+00]],
            [[-1.049999952e+00, -1.3600000143e+00,  4.3000000715e-01,  8.0000001192e-01],
             [ 6.865886449e-01, -2.7504599094e+00, -1.4547507762e+00, -1.2115331888e+00],
             [ 2.044222593e+00, -2.2286529541e+00, -2.0166194438e+00, -2.4150242805e+00],
             [ 3.183010101e+00,  1.3153430223e+00, -2.0461211204e+00, -1.9447833299e+00],
             [ 2.406511783e+00, -6.8622183799e-01,  4.3126922845e-01,  1.2989995479e+00],
             [ 5.505656242e+00, -5.6250834465e-01,  4.0067559480e-01,  2.6752333641e+00]]],
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
             [ 0.29,  0.88, -0.79,  1.48]]], device=device)
        B = torch.tensor([
            [[ 0.01, -0.99,  0.97], [-0.44, -0.10,  0.80],
             [-1.71,  2.33,  0.41], [-1.13, -0.93, -0.08]],
            [[ 0.04,  0.25,  0.12], [ 1.76,  1.42, -0.78],
             [-1.28, -0.21,  0.75], [-0.52,  0.64, -0.05]]], device=device)
        C = torch.tile(torch.eye(n_state, device=device), (n_batch, 1, 1))
        D = torch.zeros(n_batch, n_state, n_ctrl, device=device)
        c1 = torch.tensor([[ 0.25, -0.56, -0.95,  1.18],
                           [ 0.76, -0.51, -0.95,  1.18]], device=device)
        c2 = torch.zeros(n_batch, n_state, device=device)
        x_init = torch.tensor([[ 1.50, -0.34, -2.18,  0.54],
                               [-1.05, -1.36,  0.43,  0.80]], device=device)

        lti = pp.module.LTI(A, B, C, D, c1, c2).to(device)
        LQR = pp.module.LQR(lti, Q, p, T).to(device)
        x, u, cost = LQR(x_init)

        torch.testing.assert_close(x_ref, x, rtol=1e-5, atol=1e-3)
        torch.testing.assert_close(u_ref, u, atol=1e-5, rtol=1e-3)


    def test_lqr_ltv(self, device='cpu'):

        # The reference data
        x_ref = torch.tensor([
            [[ 1.5000000000e+00, -3.4000000357e-01, -2.1800000667e+00,  5.4000002145e-01],
             [ 2.3775162696e+00,  5.3751611709e-01, -1.3024839162e+00,  1.4175162315e+00],
             [ 3.9909069538e+00,  3.1090658903e-01, -3.3690934181e+00,  2.0709068775e+00],
             [ 1.0442220687e+01, -5.9778088331e-01, -1.1637781143e+01,  4.6822204589e+00],
             [ 3.9795291900e+01, -4.3647155761e+00, -4.8524715423e+01,  1.6755289077e+01],
             [ 2.0917645263e+02, -1.1623557090e+01, -2.3242353820e+02,  9.3976455688e+01]],
            [[-1.0499999523e+00, -1.3600000143e+00,  4.3000000715e-01,  8.0000001192e-01],
             [-9.9928379058e-03, -3.1999289989e-01,  1.4700071811e+00,  1.8400070667e+00],
             [-7.5955772399e-01, -1.3795578479e+00,  2.2004423141e+00,  2.9404420852e+00],
             [-3.8077919483e+00, -5.6677923202e+00,  5.0722084045e+00,  7.2922077178e+00],
             [-1.7204711914e+01, -2.4644714355e+01,  1.8315288543e+01,  2.7195285797e+01],
             [-7.5823402404e+01, -1.1302341461e+02,  1.0177659606e+02,  1.4617657470e+02]]],
            device=device)

        u_ref = torch.tensor([
            [[-6.1749362945e-01, 1.3425052165e+00,  1.5250453352e-01],
             [-1.0373514890e+00, 9.2264413833e-01, -2.6735547184e-01],
             [-1.0800623893e+00, 8.7994861602e-01, -3.1005311012e-01],
             [-1.0744695663e+00, 8.8553714752e-01, -3.0446565151e-01],
             [-2.3000000417e-01, 1.7300000190e+00,  5.4000002145e-01]],
            [[-5.6333011388e-01, 1.3966690301e+00,  2.0666818320e-01],
             [-1.0332590341e+00, 9.2673635482e-01, -2.6326334476e-01],
             [-1.0799088478e+00, 8.8010215759e-01, -3.0989956855e-01],
             [-1.0744656324e+00, 8.8554108142e-01, -3.0446171760e-01],
             [-2.3000000417e-01, 1.7300000190e+00,  5.4000002145e-01]]],
            device=device)

        n_batch, T = 2, 5
        n_state, n_ctrl = 4, 3

        Q = torch.tile(torch.eye(n_state + n_ctrl, device=device), (n_batch, 1, 1))
        p = torch.tensor([[-1.00, -0.68, -0.35, -1.42, 0.23, -1.73, -0.54],
                          [-1.00, -0.68, -0.35, -1.42, 0.23, -1.73, -0.54]], device=device)
        rt = torch.arange(1, T+1, device=device).view(T, 1, 1)
        A = rt * torch.tile(torch.eye(n_state, device=device), (n_batch, T, 1, 1))
        B = rt * torch.ones(n_batch, T, n_state, n_ctrl, device=device)
        C = torch.tile(torch.eye(n_state, device=device), (n_batch, T, 1, 1))
        D = torch.zeros(n_batch, T, n_state, n_ctrl, device=device)
        x_init = torch.tensor([[ 1.50, -0.34, -2.18,  0.54],
                               [-1.05, -1.36,  0.43,  0.80]], device=device)

        class MyLTV(pp.module.LTV):

            def __init__(self, A, B, C, D):
                super().__init__()
                self.register_buffer('_A', A)
                self.register_buffer('_B', B)
                self.register_buffer('_C', C)
                self.register_buffer('_D', D)

            def getA(self, t):
                return self._A[...,t,:,:]

            def getB(self, t):
                return self._B[...,t,:,:]

            def getC(self, t):
                return self._C[...,t,:,:]

            def getD(self, t):
                return self._D[...,t,:,:]

        ltv = MyLTV(A, B, C, D).to(device)
        lqr  = pp.module.LQR(ltv, Q, p, T).to(device)
        x, u, cost = lqr(x_init)

        torch.testing.assert_close(x_ref, x, atol=1e-5, rtol=1e-3)
        torch.testing.assert_close(u_ref, u, atol=1e-5, rtol=1e-3)


if __name__ == '__main__':
    test = TestLQR()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test.test_lqr_linear(device)
    test.test_lqr_ltv(device)
