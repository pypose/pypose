import torch
import pypose as pp

class TestMPPI:
    class Simple2DNav(pp.module.System):

        def __init__(self,dt,length=1.0):
            super().__init__()
            self._tau = dt
            self._length=length

        def state_transition(self, state, input, t=None):
            x, y, theta = state.moveaxis(-1, 0)
            v, omega = input.squeeze().moveaxis(-1, 0)

            xDot = v * torch.cos(theta)
            yDot = v * torch.sin(theta)
            thetaDot = omega

            _dstate = torch.stack((xDot, yDot, thetaDot), dim=-1)

            return (state.squeeze() + torch.mul(_dstate, self._tau)).unsqueeze(0)

        def observation(self, state, input, t=None):
            return state

    torch.manual_seed(0)
    x0 = torch.tensor([0., 0., 0.], requires_grad=False)
    dt = 0.1
    cost_fn = lambda x, u, t: (x[..., 0] - 10)**2 + (x[..., 1] - 10)**2 + (u[..., 0])**2
    mppi = pp.module.MPPI(
        dynamics=Simple2DNav(dt),
        running_cost=cost_fn,
        nx=3,
        noise_sigma=torch.eye(2) * 1,
        num_samples=100,
        horizon=5,
        lambda_=0.01
        )
    u, xn= mppi.forward(x0)
    x_ref=torch.tensor([[ 0.0000e+00,  0.0000e+00,  0.0000e+00],
            [-1.0977e-01,  0.0000e+00,  6.9991e-02],
            [-6.0990e-02,  3.4199e-03,  8.2744e-03],
            [ 7.8084e-02,  4.5706e-03, -5.6704e-02],
            [ 6.4848e-02,  5.3220e-03, -8.1209e-02],
            [ 1.3131e-01, -8.6915e-05, -1.8065e-01]])
    u_ref=torch.tensor([[-1.0977,  0.6999],
            [ 0.4890, -0.6172],
            [ 1.3908, -0.6498],
            [-0.1326, -0.2450],
            [ 0.6668, -0.9944]])
    torch.testing.assert_close(x_ref, xn, atol=1e-5, rtol=1e-3)
    torch.testing.assert_close(u_ref, u, atol=1e-5, rtol=1e-3)

if __name__ == '__main__':
    test = TestMPPI()
