from cmath import inf
import numpy as np
import torch as torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pypose as pp

from dynamics import LorenzAttractor

if __name__=="__main__":
    # Create Lorenz attractor dynamics
    IC = np.array([-8,8,27])
    # Constant declarations
    sigma = 10
    rho = 28
    beta = 8/3
    # Derivative function
    def f(t,x):
        xdot = sigma*(x[1]-x[0])
        ydot = x[0]*(rho-x[2])-x[1]
        zdot = x[0]*x[1]-beta*x[2]
        return xdot,ydot,zdot
    # Create time span for trajectory prediction
    tspan = [0.001,10]                                # Simluation time span
    num_tsteps = 1000                                # Number of time steps
    t_eval = np.linspace(tspan[0],tspan[1],num_tsteps) # Vector of all times
    t_step = t_eval[1] - t_eval[0]                     # Find time step size
    # Solve for trajectory
    sol = solve_ivp(f,tspan,IC,method='DOP853',t_eval=t_eval,vectorized=True,rtol=1e-12,atol=1e-12)
    sol = torch.tensor(sol.y).float()

    stateTransition = nn.Sequential(nn.Linear(3,5),
                                    nn.ReLU(inplace=False),
                                    nn.Linear(5,5),
                                    nn.ReLU(inplace=False),
                                    nn.Linear(5,3))
    
    lorenzSolver = LorenzAttractor(t_step,stateTransition)

    pred = torch.zeros(np.shape(sol))
    state = torch.tensor(IC).float()
    pred[:,0] = state;

    stateTransitionOptim = torch.optim.Adam(stateTransition.parameters(),lr=0.01)

    loss = torch.inf
    while loss > 0.01:
        for i in range(1,num_tsteps):
            next_state = lorenzSolver.forward(state,0)
            pred[:,i] = next_state 
            state = next_state
        loss = lorenzSolver.loss(sol,pred)
        loss.backward(retain_graph=True)
        stateTransitionOptim.step()

    print("done")