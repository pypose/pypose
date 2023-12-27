import torch
import numpy as np
import pypose as pp
from tqdm import tqdm
import matplotlib.pyplot as plt

dt = 0.1

class Bicycle(pp.module.NLS):
    def __init__(self):
        super().__init__()

    def state_transition(self, state, input, t=None):
        v, omega = input[..., 0], input[..., 1]
        c, s = state[..., 2], state[..., 3]
        theta = torch.atan2(s, c) + omega * dt

        x = state[..., 0] + v * c * dt
        y = state[..., 1] + v * s * dt
        c, s = torch.cos(theta), torch.sin(theta)
        return torch.stack([x, y, c, s], dim=-1)
    
    def observation(self, state, input, t=None):
        return state

def visualize_X(X, waypoints=None):
    X = torch.stack(X).detach().numpy()
    plt.plot(X[:, 0], X[:, 1], 'o-', label='trajectory', zorder = 1, alpha = 0.5)

    waypoints = waypoints.detach().numpy()
    if waypoints is not None:
        plt.scatter(waypoints[:, 0], waypoints[:, 1], color = "red", marker = "o", s = 100, label='waypoints', zorder = 3)
        for waypoint in waypoints:
            theta = np.arctan2(waypoint[3], waypoint[2])
            plt.arrow(waypoint[0], waypoint[1], 1*np.cos(theta), 1*np.sin(theta), color='black', width=0.1, zorder = 2)
    plt.axis('equal')
    plt.title('Bicycle trajectory')
    plt.legend()
    plt.show()

def main():
    dynamics = Bicycle()
    T, max_iter = 7, 50
    dt = 0.1
    r = 5
    # thetas = torch.tensor([torch.pi/2, 0, -torch.pi/2, -torch.pi])
    thetas = torch.tensor([torch.pi/2, torch.pi/4, 0, -torch.pi/4, -torch.pi/2, -torch.pi*3/4, -torch.pi, torch.pi*3/4, torch.pi/2, 
                           torch.pi*3/4, torch.pi, -torch.pi*3/4, -torch.pi/2, -torch.pi/4, 0, torch.pi/4, torch.pi/2])
    # waypoints = \
    #     torch.tensor([[
    #         [0, 0, thetas[0].cos(), thetas[0].sin()],
    #         [r, r, thetas[1].cos(), thetas[1].sin()],
    #         [2*r, 0, thetas[2].cos(), thetas[2].sin()],
    #         [r, -r, thetas[3].cos(), thetas[3].sin()],
    #         [0, 0, thetas[0].cos(), thetas[0].sin()]]], requires_grad=True)
    sqrt2 = 2**0.5
    waypoints = \
        torch.tensor([[
            [0, 0, thetas[0].cos(), thetas[0].sin()],
            [r-r/sqrt2, r/sqrt2, thetas[1].cos(), thetas[1].sin()],
            [r, r, thetas[2].cos(), thetas[2].sin()],
            [r+r/sqrt2, r/sqrt2, thetas[3].cos(), thetas[3].sin()],
            [2*r, 0, thetas[4].cos(), thetas[4].sin()],
            [r+r/sqrt2, -r/sqrt2, thetas[5].cos(), thetas[5].sin()],
            [r, -r, thetas[6].cos(), thetas[6].sin()],
            [r-r/sqrt2, -r/sqrt2, thetas[7].cos(), thetas[7].sin()],
            [0, 0, thetas[8].cos(), thetas[8].sin()],
            [r/sqrt2-r, r/sqrt2, thetas[9].cos(), thetas[9].sin()],
            [-r, r, thetas[10].cos(), thetas[10].sin()],
            [-r/sqrt2-r, r/sqrt2, thetas[11].cos(), thetas[11].sin()],
            [-2*r, 0, thetas[12].cos(), thetas[12].sin()],
            [-r/sqrt2-r, -r/sqrt2, thetas[13].cos(), thetas[13].sin()],
            [-r, -r, thetas[14].cos(), thetas[14].sin()],
            [r/sqrt2-r, -r/sqrt2, thetas[15].cos(), thetas[15].sin()],
            [0, 0, thetas[0].cos(), thetas[0].sin()]]], requires_grad=True)

    n_batch = 1
    n_state, n_ctrl = 4, 2
    Q = torch.tile(torch.eye(n_state + n_ctrl), (n_batch, T, 1, 1))
    Q[..., 0, 0], Q[..., 1, 1], Q[..., 2, 2], Q[..., 3, 3]= 0.01, 0.01, 0.01, 0.01
    Q[..., 4, 4], Q[..., 5, 5] = 1, 1
    Q[:, -1, 0, 0], Q[:, -1, 1, 1], Q[:, -1, 2, 2], Q[:, -1, 3, 3] = 100, 100, 10, 10
    p = torch.tile(torch.zeros(n_state + n_ctrl), (n_batch, T, 1))

    stepper = pp.utils.ReduceToBason(steps=10, verbose=False)
    MPC = pp.module.MPC(dynamics, Q, p, T, stepper=stepper)

    xt = waypoints[:, 0, :]
    X, U = [xt.squeeze()], []
    print("The intial point is", waypoints[:, 0, :])

    for pt in range(1, waypoints.shape[1]):
        print("The target point is", waypoints[:, pt, :])
        target = waypoints[:, pt, :]
        
        for t in tqdm(range(max_iter)):
            # dx = target - xt
            _, u_mpc, _ = MPC(dt, xt, target)
            u_mpc = torch.clamp(u_mpc, -5, 5)
            
            xt = dynamics(xt, u_mpc[...,0,:])[0]
            X.append(xt.squeeze())
            U.append(u_mpc[...,0,:].squeeze())
            if torch.norm(xt[:,0:2] - target[:,0:2]) < 0.1:
                print("The target point is reached")
                break
    visualize_X(X, waypoints.squeeze())

if __name__ == "__main__":
    main()
