import torch, pypose as pp
import math, matplotlib.pyplot as plt


class Floquet(pp.module.NLS):
    '''
    Floquet system is periodic and time-varying.
    '''
    def __init__(self):
        super().__init__()

    def state_transition(self, state, input, t):
        cc = (2 * math.pi * t / 100).cos()
        ss = (2 * math.pi * t / 100).sin()
        A = torch.tensor([[   1.,  cc/10],
                          [cc/10,     1.]], device=t.device)
        B = torch.tensor([[ss],
                          [1.]], device=t.device)

        return A @ state + B @ input

    def observation(self, state, input, t):
        return state + t


def subPlot(ax, x, y, xlabel=None, ylabel=None):
    x = x.detach().cpu().numpy()
    y = y.detach().cpu().numpy()
    ax.plot(x, y)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    N = 100    # Number of time steps
    time  = torch.arange(0, N, device=device)
    input = (2 * math.pi * time / 50).sin()

    state = torch.zeros(N, 2, device=device)
    state[0] = torch.tensor([1., 1.], device=device)
    obser = torch.zeros(N, 2, device=device)

    # Calculate trajectory
    model = Floquet().to(device)
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
    plt.show()
    