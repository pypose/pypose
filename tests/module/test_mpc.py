import torch, pypose as pp


class TestMPC:

    def test_ilqr_cartpole(self, device='cpu'):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # The reference data
        x_ref = torch.tensor([
            [[ 0.0000000000e+00,  0.0000000000e+00, 3.1415927410e+00,  0.0000000000e+00],
             [ 0.0000000000e+00,  7.8299985034e-05, 3.1415927410e+00,  3.9145703340e-05],
             [ 7.8299984807e-07, -1.5641693607e-04, 3.1415932178e+00, -7.8217039117e-05],
             [-7.8116948998e-07, -3.8382178172e-04, 3.1415925025e+00, -1.9194713968e-04],
             [-4.6193872549e-06, -7.2401645593e-04, 3.1415905952e+00, -3.6203706986e-04],
             [-1.1859551705e-05, -7.6239358168e-04, 3.1415870189e+00, -3.8112464244e-04]]],
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
                self.poleml = self.polemass * self.length
                self.totalMass = self.cartmass + self.polemass

            def state_transition(self, state, input, t=None):
                x, xDot, theta, thetaDot = state.squeeze()
                force = input.squeeze()
                costheta = torch.cos(theta)
                sintheta = torch.sin(theta)

                temp = (force + self.poleml * thetaDot**2 * sintheta) / self.totalMass
                thetaAcc = (self.gravity * sintheta - costheta * temp) / \
                    (self.length * (4 / 3 - self.polemass * costheta**2 / self.totalMass))
                xAcc = temp - self.poleml * thetaAcc * costheta / self.totalMass
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
        stepper = pp.utils.ReduceToBason(steps=10, verbose=False)
        MPC = pp.module.MPC(cartPoleSolver, Q, p, T, stepper=stepper).to(device)
        x, u, cost = MPC(dt, x_init, u_init=current_u)

        torch.testing.assert_close(x_ref, x, atol=1e-5, rtol=1e-3)
        torch.testing.assert_close(u_ref, u, atol=1e-5, rtol=1e-3)


if __name__ == '__main__':
    test = TestMPC()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test.test_ilqr_cartpole(device)
