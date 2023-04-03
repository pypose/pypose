import argparse, os
import torch, pypose as pp
import matplotlib.pyplot as plt


class NNDynamics(pp.module.NLS):
    def __init__(self, hidden):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(2, hidden[0]),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden[0], hidden[1]),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden[1], 2))

    def state_transition(self, state, input, t=None):
        return self.net(state) + input

    def observation(self, state, input, t=None):
        return state


def subPlot(ax, x, y, xlabel=None, ylabel=None):
    x = x.detach().cpu().numpy()
    y = y.detach().cpu().numpy()
    ax.plot(x, y)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='NeuralNet Example')
    parser.add_argument("--device", type=str, default='cpu', help="cuda or cpu")
    parser.add_argument("--save", type=str, default='./examples/module/dynamics/save/', 
                        help="location of png files to save")
    parser.add_argument('--show', dest='show', action='store_true',
                        help="show plot, default: False")
    parser.set_defaults(show=False)
    args = parser.parse_args(); print(args)
    os.makedirs(os.path.join(args.save), exist_ok=True)

    dt = 0.01  # Time step size
    N  = 1000  # Number of time steps
    time  = torch.arange(0, N, device=args.device) * dt
    input = torch.sin(time)
    state = torch.zeros(N, 2, device=args.device) # trajectory
    state[0] = torch.tensor([1., 1.], device=args.device)

    model = NNDynamics([5, 10]).to(args.device)
    for i in range(N - 1):
        state[i + 1], _ = model(state[i], input[i])

    f, ax = plt.subplots(nrows=2, sharex=True)
    subPlot(ax[0], time, state[:, 0], ylabel='X')
    subPlot(ax[1], time, state[:, 1], ylabel='Y', xlabel='Time')

    figure = os.path.join(args.save + 'neuralnet.png')
    plt.savefig(figure)
    print("Saved to", figure)

    if args.show:
        plt.show()
