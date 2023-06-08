import torch, pypose as pp


class TestMPC:

    def test_ilqr_cartpole(self, device='cpu'):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # The reference data
        x_ref = torch.tensor([
            [[ 0.0000000000000000e+00,  0.0000000000000000e+00, 3.1415927410125732e+00,  0.0000000000000000e+00],
             [ 0.0000000000000000e+00,  7.8299985034391284e-05, 3.1415927410125732e+00,  3.9145703340182081e-05],
             [ 7.8299984807017609e-07, -1.5641693607904017e-04, 3.1415932178497314e+00, -7.8217039117589593e-05],
             [-7.8116948998285807e-07, -3.8382178172469139e-04, 3.1415925025939941e+00, -1.9194713968317956e-04],
             [-4.6193872549338266e-06, -7.2401645593345165e-04, 3.1415905952453613e+00, -3.6203706986270845e-04],
             [-1.1859551705128979e-05, -7.6239358168095350e-04, 3.1415870189666748e+00, -3.8112464244477451e-04]]],
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
        current_u = torch.sin(time).unsqueeze(1).unsqueeze(0)

        x_init = torch.tensor([[0, 0, torch.pi, 0]], device=device)

        cartPoleSolver = CartPole(dt, len, m_cart, m_pole, g).to(device)
        MPC = pp.module.MPC(cartPoleSolver, T, step=15).to(device)
        x, u, cost = MPC(Q, p, x_init, dt, current_u=current_u)

        torch.testing.assert_close(x_ref, x, atol=1e-5, rtol=1e-3)
        torch.testing.assert_close(u_ref, u, atol=1e-5, rtol=1e-3)


if __name__ == '__main__':
    test = TestMPC()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test.test_ilqr_cartpole(device)
