import pypose as pp
import torch as torch


def test_nlqr_cartpole():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # The reference data
    x_ref = torch.tensor([
        [[ 0.0000000000000000e+00,  0.0000000000000000e+00, 3.1415927410125732e+00,  0.0000000000000000e+00],
         [ 0.0000000000000000e+00,  7.8318604209925979e-05, 3.1415927410125732e+00,  3.9155012927949429e-05],
         [ 7.8318601026694523e-07, -1.5194457955658436e-04, 3.1415932178497314e+00, -7.5980868132319301e-05],
         [-7.3625972163426923e-07, -3.7045811768621206e-04, 3.1415925025939941e+00, -1.8526532221585512e-04],
         [-4.4408407120499760e-06, -6.9732154952362180e-04, 3.1415905952453613e+00, -3.4868964576162398e-04]]],
        device=device)

    u_ref = torch.tensor([[[1.7622329294681549e-01],
                           [-5.1808577775955200e-01],
                           [-4.9161392450332642e-01],
                           [-7.3545390367507935e-01],
                           [-4.6510662883520126e-02]]],device=device)

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

    NLQR = pp.module.NLQR(cartPoleSolver, Q, p, T).to(device)
    x, u, cost = NLQR(x_init, current_x, current_u, time)

    torch.testing.assert_close(x_ref, x, atol=1e-5, rtol=1e-3)
    torch.testing.assert_close(u_ref, u, atol=1e-5, rtol=1e-3)


if __name__ == '__main__':
    test_nlqr_cartpole()