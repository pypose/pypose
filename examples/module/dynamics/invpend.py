import argparse, os
import torch, pypose as pp
import math, matplotlib.pyplot as plt

class InvPend(pp.module.NLS):
    '''
    # Inverted-pendulum dynamics
    '''
    def __init__(self, dt, length=[10.0], gravity=10.0):
        super(InvPend, self).__init__(xdim=2, udim=1, ydim=2)
        self.tau = dt
        self.length = length
        self.gravity = gravity

    def state_transition(self, state, input, t=None):
        force = input.squeeze(-1)
        _dstate = torch.stack([state[...,1], force+self.gravity/self.length[0]*torch.sin(state[...,0].clone())], dim=-1)
        return state + torch.mul(_dstate, self.tau)

    def observation(self, state, input, t=None):
        return state


def subPlot(ax, x, y, xlabel=None, ylabel=None):
    x = x.detach().cpu().numpy()
    y = y.detach().cpu().numpy()
    ax.plot(x, y)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='InvPend Example')
    parser.add_argument("--device", type=str, default='cpu', help="cuda or cpu")
    parser.add_argument("--save", type=str, default='./examples/module/dynamics/save/',
                        help="location of png files to save")
    parser.add_argument('--show', dest='show', action='store_true',
                        help="show plot, default: False")
    parser.set_defaults(show=False)
    args = parser.parse_args(); print(args)
    os.makedirs(os.path.join(args.save), exist_ok=True)

    N = 100    # Number of time steps
    time  = torch.arange(0, N, device=args.device)
    input = (2 * math.pi * time / 50).sin()

    state = torch.zeros(N, 2, device=args.device)
    state[0] = torch.tensor([1., 1.], device=args.device)
    obser = torch.zeros(N, 2, device=args.device)

    # Calculate trajectory
    dt = 0.05   # Delta t
    model = InvPend(dt).to(args.device)
    for i in range(N - 1):
        state[i + 1], obser[i] = model(state[i], input[i])

    # Jacobian computation - Find jacobians at the last step
    vars = ['A', 'B', 'C', 'D', 'c1', 'c2']
    model.set_refpoint()
    [print(v, getattr(model, v)) for v in vars]

    # Jacobian computation - Find jacobians at the 5th step
    idx = 5
    model.set_refpoint(state=state[idx], input=input[idx], t=time[idx])
    [print(v, getattr(model, v)) for v in vars]

    # Create time plots to show dynamics
    f, ax = plt.subplots(nrows=4, sharex=True)
    subPlot(ax[0], time, state[:, 0], ylabel='State[0]')
    subPlot(ax[1], time, state[:, 1], ylabel='State[1]')
    subPlot(ax[2], time[:-1], obser[:-1, 0], ylabel='Observe[0]')
    subPlot(ax[3], time[:-1], obser[:-1, 1], ylabel='Observe[1]', xlabel='Time')

    figure = os.path.join(args.save + 'invpend.png')
    plt.savefig(figure)
    print("Saved to", figure)

    if args.show:
        plt.show()
