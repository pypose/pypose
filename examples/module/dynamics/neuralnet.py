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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dt = 0.01  # Time step size
    N  = 1000  # Number of time steps
    time  = torch.arange(0, N, device=device) * dt
    input = torch.sin(time)
    state = torch.zeros(N, 2, device=device) # trajectory
    state[0] = torch.tensor([1., 1.], device=device)

    model = NNDynamics([5, 10]).to(device)
    for i in range(N - 1):
        state[i + 1], _ = model(state[i], input[i])

    f, ax = plt.subplots(nrows=2, sharex=True)
    subPlot(ax[0], time, state[:, 0], ylabel='X')
    subPlot(ax[1], time, state[:, 1], ylabel='Y', xlabel='Time')
    plt.show()
