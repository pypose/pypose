import torch, pypose as pp
import math, matplotlib.pyplot as plt
from pypose.module.dynamics import System


# Create class for cart-pole dynamics
class CartPole(System):
    def __init__(self, dt, length, cartmass, polemass, gravity):
        super(CartPole, self).__init__()
        self.tau = dt
        self.length = length
        self.cartmass = cartmass
        self.polemass = polemass
        self.gravity = gravity
        self.polemassLength = self.polemass * self.length
        self.totalMass = self.cartmass + self.polemass

    def state_transition(self, state, input, t=None):
        x, xDot, theta, thetaDot = state
        force = input.squeeze()
        costheta = theta.cos()
        sintheta = theta.sin()

        temp = (force + self.polemassLength * thetaDot**2 * sintheta) / self.totalMass

        thetaAcc = (self.gravity * sintheta - costheta * temp) / \
            (self.length * (4.0 / 3.0 - self.polemass * costheta**2 / self.totalMass))
    
        xAcc = temp - self.polemassLength * thetaAcc * costheta / self.totalMass

        _dstate = torch.stack((xDot, xAcc, thetaDot, thetaAcc))

        return state + _dstate * self.tau

    def observation(self, state, input, t=None):
        return state


def subPlot(ax, x, y, xlabel=None, ylabel=None):
    x = x.detach().cpu().numpy()
    y = y.detach().cpu().numpy()
    ax.plot(x, y)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Create parameters for cart pole trajectory
    dt = 0.01   # Delta t
    len = 1.5   # Length of pole
    m_cart = 20 # Mass of cart
    m_pole = 10 # Mass of pole
    g = 9.81    # Accerleration due to gravity
    N = 1000    # Number of time steps

    # Time and input
    time  = torch.arange(0, N, device=device) * dt
    input = torch.sin(time)
    state = torch.zeros(N, 4, dtype=float, device=device)
    state[0] = torch.tensor([0, 0, math.pi, 0], dtype=float, device=device)

    # Create dynamics solver object
    model = CartPole(dt, len, m_cart, m_pole, g).to(device)
    for i in range(N - 1):
        state[i + 1], _ = model(state[i], input[i])

    # Jacobian computation - Find jacobians at the last step
    model.set_refpoint(state=state[-1,:], input=input[-1], t=time[-1])
    vars = ['A', 'B', 'C', 'D', 'c1', 'c2']
    [print(v, getattr(model, v)) for v in vars]

    # Create time plots to show dynamics
    f, ax = plt.subplots(nrows=4, sharex=True)
    x, xdot, theta, thetadot = state.T
    subPlot(ax[0], time, x, ylabel='X')
    subPlot(ax[1], time, xdot, ylabel='X dot')
    subPlot(ax[2], time, theta, ylabel='Theta')
    subPlot(ax[3], time, thetadot, ylabel='Theta dot', xlabel='Time')
    plt.show()
