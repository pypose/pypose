from pypose.module.dynamics import System
import math
import torch as torch
import numpy as np
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# We consider a Floquet system, which is periodic and an example of time-varying systems
class Floquet(System):
    def __init__(self):
        super(Floquet, self).__init__()

    def state_transition(self, state, input, t):
        cc = torch.cos(2*math.pi*t/100)
        ss = torch.sin(2*math.pi*t/100)
        A = torch.tensor([
            [1., cc/10],
            [cc/10, 1.]])
        B = torch.tensor([
            [ss],
            [1.]])
        return (state.matmul(A) + B.matmul(input)).squeeze()

    def observation(self, state, input, t):
        return state + t

def createTimePlot(x, y, figname="Un-named plot", title=None, xlabel=None, ylabel=None):
    f = plt.figure(figname)
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    return f

if __name__ == "__main__":
    N = 100    # Number of time steps

    # Time, Input, Initial state
    time  = torch.arange(0, N+1)
    input = torch.sin(2*math.pi*time/50)
    state = torch.tensor([1., 1.])

    # Create dynamics solver object
    solver = Floquet()

    # Calculate trajectory
    state_all = torch.zeros(N+1, 2)
    state_all[0] = state
    obser_all = torch.zeros(N, 2)

    for i in range(N):
        state_all[i+1], obser_all[i] = solver(state_all[i], input[i])

    # Create time plots to show dynamics
    f, ax = plt.subplots(nrows=4, sharex=True)
    for _i in range(2):
        ax[_i].plot(time, state_all[:,_i], label='pp')
        ax[_i].set_ylabel(f'State {_i}')
    for _i in range(2):
        ax[_i+2].plot(time[:-1], obser_all[:,_i], label='pp')
        ax[_i+2].set_ylabel(f'Observation {_i}')
    ax[-1].set_xlabel('time')
    ax[-1].legend()

    # Jacobian computation - Find jacobians at the last step
    vars = ['A', 'B', 'C', 'D', 'c1', 'c2']
    solver.set_refpoint()
    [print(_v, getattr(solver, _v)) for _v in vars]

    # Jacobian computation - Find jacobians at the 5th step
    idx = 5
    solver.set_refpoint(state=state_all[idx], input=input[idx], t=time[idx])
    [print(_v, getattr(solver, _v)) for _v in vars]

    plt.show()
    