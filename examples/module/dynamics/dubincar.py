import argparse, os
import torch, pypose as pp
import math, matplotlib.pyplot as plt


class DubinCar(pp.module.NLS):
    def __init__(self, dt):
        self.tau = dt
        super(DubinCar, self).__init__()

    # Use RK4 to infer the k+1 state
    def state_transition(self, state, input, t=None):
        return self.rk4(state, input, self.tau)

    def rk4(self, state, input, t=None):
        k1 = self.xdot(state, input)
        k1_state = state + k1 * t / 2

        k2 = self.xdot(k1_state, input)
        k2_state = state + k2 * t / 2

        k3 = self.xdot(k2_state, input)
        k3_state = state + k3 * t

        k4 = self.xdot(k3_state, input)

        return state + (k1 + 2 * k2 + 2 * k3 + k4) / 6 * t

    def observation(self, state, input, t=None):
        return state

    def xdot(self, state, input):
        orientation, vel, w = state[2:]
        # acceleration and angular acceleration
        acc, w_dot = input

        return torch.stack(
            [
                vel * torch.cos(orientation),
                vel * torch.sin(orientation),
                w,
                acc,
                w_dot
            ]
        )


def subPlot(ax, x, y, xlabel=None, ylabel=None):
    x = x.detach().cpu().numpy()
    y = y.detach().cpu().numpy()
    ax.plot(x, y)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dubincar Example')
    parser.add_argument("--device", type=str, default='cpu', help="cuda or cpu")
    parser.add_argument("--save", type=str, default='./examples/module/dynamics/save/',
                        help="location of png files to save")
    parser.add_argument('--show', dest='show', action='store_true',
                        help="show plot, default: False")
    parser.set_defaults(show=False)
    args = parser.parse_args(); print(args)
    os.makedirs(os.path.join(args.save), exist_ok=True)

    N = 100    # Number of time steps
    dt = 0.1
    state = torch.zeros(N, 5, device=args.device)
    time  = torch.arange(0, N, device=args.device) * dt
    input = torch.ones(N, 2, device=args.device)

    model = DubinCar(dt).to(args.device)
    # Calculate trajectory
    for i in range(N - 1):
        state[i + 1], _ = model(state[i], input[i])

    # Create time plots to show dynamics
    f, ax = plt.subplots(nrows=3, sharex=True)
    subPlot(ax[0], time, state[:, 0], ylabel='X position (m)')
    subPlot(ax[1], time, state[:, 1], ylabel='Y position (m)')
    subPlot(ax[2], time, state[:, 2] / torch.pi * 180, ylabel='Orientation (degree)')

    figure = os.path.join(args.save + 'dubincar.png')
    plt.savefig(figure)
    print("Saved to", figure)

    if args.show:
        plt.show()
