from pypose.module.dynamics import System
import torch as torch
import numpy as np
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create class for cart-pole dynamics
class CartPole(System):
    def __init__(self,dt,length,cartmass,polemass,gravity):
        super(CartPole, self).__init__(time=False)
        self._tau = dt
        self._length = length
        self._cartmass = cartmass
        self._polemass = polemass
        self._gravity = gravity
        self._polemassLength = self._polemass*self._length
        self._totalMass = self._cartmass + self._polemass

    def state_transition(self,state,input):
        x,xDot,theta,thetaDot = state
        force = input.squeeze()
        costheta = torch.cos(theta)
        sintheta = torch.sin(theta)

        temp = (
            force + self._polemassLength * thetaDot**2 * sintheta
        ) / self._totalMass
        thetaAcc = (self._gravity * sintheta - costheta * temp) / (
            self._length * (4.0 / 3.0 - self._polemass * costheta**2 / self._totalMass)
        )
        xAcc = temp - self._polemassLength * thetaAcc * costheta / self._totalMass

        _dstate = torch.stack((xDot,xAcc,thetaDot,thetaAcc))

        return state+torch.mul(_dstate,self._tau)
    
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
    # Create parameters for cart pole trajectory
    dt = 0.01   # Delta t
    len = 1.5   # Length of pole
    m_cart = 20 # Mass of cart
    m_pole = 10 # Mass of pole
    g = 9.81    # Accerleration due to gravity
    N = 1000    # Number of time steps

    # Time and input
    time  = torch.arange(0,N+1) * dt
    input = torch.sin(time)
    # Initial state
    state = torch.tensor([0,0,np.pi,0],dtype=float)

    # Create dynamics solver object
    cartPoleSolver = CartPole(dt,len,m_cart,m_pole,g)

    # Calculate trajectory
    state_all = torch.zeros(N+1,4,dtype=float)
    state_all[0,:] = state
    for i in range(N):
        state_all[i+1], _ = cartPoleSolver.forward(state_all[i],input[i])

    # Create time plots to show dynamics
    x,xdot,theta,thetadot = state_all.T
    x_fig = createTimePlot(time,x,figname="x Plot",xlabel="Time",ylabel="x",title="x Plot")
    xdot_fig = createTimePlot(time,xdot,figname="x dot Plot",xlabel="Time",ylabel="x dot",title="x dot Plot")
    theta_fig = createTimePlot(time,theta,figname="theta Plot",xlabel="Time",ylabel="theta",title="theta Plot")
    thetadot_fig = createTimePlot(time,thetadot,figname="theta dot Plot",xlabel="Time",ylabel="theta dot",title="theta dot Plot")

    # Jacobian computation - Find jacobians at the last step
    jacob_state, jacob_input = state_all[-1,:].T, input[-1]
    cartPoleSolver.set_linearization_point(jacob_state,jacob_input.unsqueeze(0),time[-1])
    A = (cartPoleSolver.A).numpy()
    B = (cartPoleSolver.B).numpy()
    C = (cartPoleSolver.C).numpy()
    D = (cartPoleSolver.D).numpy()
    c1 = (cartPoleSolver.c1).numpy()
    c2 = (cartPoleSolver.c2).numpy()

    torch.save([state_all,A,B,C,D,c1,c2], 'cartpole_data.pt')

    plt.show()
    