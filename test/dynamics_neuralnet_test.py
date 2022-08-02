import pypose.module.dynamics as ppmd
import torch as torch
import numpy as np
import matplotlib.pyplot as plt

class neuralnetDynamics(ppmd._System):
    def __init__(self, func, time=False):
        super().__init__(time)
        self.net = func
    
    def state_transition(self,state,input):
        return self.net(state)+input
    
    def observation(self,state,input):
        return state

def createTimePlot(x,y,figname="Un-named plot",title=None,xlabel=None,ylabel=None):
    f = plt.figure(figname)
    plt.plot(x,y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    return f

if __name__ == "__main__":
    N = 1000  # Number of points to calculate
    dt = 0.01 # Change in time
    input = torch.sin(torch.linspace(0,N*dt+1,N+1)) # Input vector
    state0 = torch.tensor((0,0))                    # Initial state
    state_all = torch.zeros(N,2).float()            # Store all states
    state_all[0,:] = state0                         # Store initial state
    # Neural network for finding next state
    func = torch.nn.Sequential(
            torch.nn.Linear(2,5),
            torch.nn.ReLU(),
            torch.nn.Linear(5,10),
            torch.nn.ReLU(),
            torch.nn.Linear(10,2)
    )
    # Create solver object
    nnSolver = neuralnetDynamics(func)
    # Calculate trajectory
    for i in range(1,N):
        state_all[i],_ = nnSolver.forward(state_all[i-1],input[i-1])
    
    # Create plots
    x,y = (state_all.T).detach().numpy()
    time = torch.linspace(0,N*dt,N)
    x_fig = createTimePlot(time,x,figname="x Plot",xlabel="Time",ylabel="x",title="x Plot")
    y_fig = createTimePlot(time,y,figname="y Plot",xlabel="Time",ylabel="y",title="y Plot")
    if 1:
        plt.show()