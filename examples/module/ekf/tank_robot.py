import math
import pypose as pp
import torch as torch
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TankRobot(pp.module.System):
    def __init__(self, dt, Q, R):
        super(TankRobot, self).__init__()
        self.state = torch.tensor([0.0, 0.0, 0.0])
        self.input = torch.tensor([0.0, 0.0])
        self.dt = dt
        self.register_buffer("Q", torch.eye(self.A.shape) if Q is None else Q)
        self.register_buffer("R", torch.eye(self.C.shape[0]) if R is None else R)
        self.set_refpoint()

    def state_transition(self, state, input, t=None):
        # input is torch.tensor([v, w]) where v is forward velo and w is angular velo
        input_dt = torch.mul(input, self.dt)
        theta = state[2] + input_dt[1]
        v_x = input_dt[0] * torch.cos(theta)
        v_y = input_dt[0] * torch.sin(theta)
        w = torch.randn(state.shape) @ self.Q  # process noise
        state = torch.tensor([state[0] + v_x, state[1] + v_y, theta]) + w
        return state

    def observation(self, state, input, t=None):
        y = state  # observation is just state here
        v = torch.randn(state.shape) @ self.R  # measurement noise
        return y + v


def createTimePlot(x, y, figname="Un-named plot", title=None, xlabel=None, ylabel=None):
    f = plt.figure(figname)
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    return f


def color_fade(c1, c2, mix=0):  # lerp from color c1 (at mix=0) to c2 (mix=1)
    return [c1[i] * mix + c2[i] * (1 - mix) for i in range(4)]


if __name__ == "__main__":

    x = 0.0  # starting x position
    y = 0.0  # starting y position
    theta = 0.0  # starting heading
    state = torch.tensor([x, y, theta])
    N = 100  # number of timesteps
    dt = 1
    time = torch.arange(0, N + 1) * dt

    est_state = torch.tensor([10.0, 0.0, 0.0])  # initial state estimate
    P = torch.eye(3)  # initial state covariance

    Q = 0.2 * torch.eye(3) + 0.1 * torch.randn(3, 3)
    R = 0.2 * torch.eye(3) + 0.1 * torch.randn(3, 3)
    model = TankRobot(dt, Q, R)  # system model
    filter = pp.module.EKF(model)  # Kalman Filter to estimate the model

    # keep history of following values
    inputs = []
    states = []
    estimates = []
    covariances = []

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for i in time:
        print("timestep {}".format(i))
        u = torch.tensor([1, 0]) + 0.1 * torch.randn(2)  # random input

        inputs.append(u)
        states.append(state)  # real state
        estimates.append(est_state)  # estimated state
        covariances.append(P)

        x = state[0]
        y = state[1]
        theta = state[2]

        state, observation = model(state, u)  # model measurement
        model.set_refpoint(state=state, input=u)
        est_state, P = filter(est_state, observation, P, u)

        color_path = tuple(color_fade([1, 0, 0, 1], [0, 0, 1, 1], mix=(i / N).item()))

        arrow = plt.quiver(
            x,
            y,
            state[0] - x,
            state[1] - y,
            scale_units="xy",
            angles="xy",
            scale=1,
            color=color_path,
        )
        ax.add_artist(arrow)

        dot = plt.Circle(
            (est_state[0], est_state[1]),
            math.hypot(P[0][0], P[1][1]),
            fill=False,
            color=color_path,
        )
        ax.add_artist(dot)

    plt.show()
