import torch
import pypose as pp
import matplotlib.pyplot as plt

def getse3(xyzrpy):
    xyz = xyzrpy[..., :3]
    rpy = xyzrpy[..., 3:]
    q = pp.euler2SO3(rpy).tensor()
    return pp.SE3(torch.cat([xyz, q], dim=-1)).Log()

class anySE3(pp.module.NLS):
    def __init__(self, dt=None):
        super().__init__()
        self.ref_traj = None
        self.dt = dt

    def state_transition(self, state, input, t):
        if self.ref_traj is None:
            return state.Retr(pp.se3(input))
        else:
            ref_SE3 = self.ref_traj[...,t,:]
            next_ref_SE3 = self.ref_traj[...,t+1,:]
            next_SE3 = (ref_SE3*state).Retr(pp.se3(input))
            return next_ref_SE3.Inv()*next_SE3
            # next_SE3 = (ref_SE3@state).Retr(pp.se3(input))
            # return next_ref_SE3.Inv()@next_SE3

    def observation(self, state, input, t=None):
        return state

    def set_reftrajectory(self, ref_traj):
        self.ref_traj = ref_traj

    def recover_dynamics(self):
        self.ref_traj = None

def visualize_traj(traj, ref_traj):
    traj, ref_traj = pp.SE3(traj), pp.SE3(ref_traj)
    x = traj.translation()[...,0].squeeze().detach().numpy()
    y = traj.translation()[...,1].squeeze().detach().numpy()
    z = traj.translation()[...,2].squeeze().detach().numpy()
    x_ref = ref_traj.translation()[...,0].squeeze().detach().numpy()
    y_ref = ref_traj.translation()[...,1].squeeze().detach().numpy()
    z_ref = ref_traj.translation()[...,2].squeeze().detach().numpy()

    plt.figure()
    plt.axes().set_aspect('equal')
    plt.plot(x, y, 'o-', label='trajectory', alpha = 0.8)
    plt.plot(x_ref, y_ref, 'o-', label='reference trajectory', alpha = 0.1)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Bicycle trajectory')
    plt.legend()
    plt.show()

    ## Plot in 3D
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot(x, y, z, 'o-', label='trajectory', alpha = 0.8)
    # ax.plot(x_ref, y_ref, z_ref, 'o-', label='reference trajectory', alpha = 0.1)
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('z')
    # ax.set_title('Bicycle trajectory')
    # ax.legend()
    # plt.show()

def waypoints_configs():
    r = 6

    sqrt2 = 2**0.5
    thetas = torch.tensor([torch.pi/2, torch.pi/4, 0, -torch.pi/4, -torch.pi/2, -torch.pi*3/4, -torch.pi, torch.pi*3/4, torch.pi/2,
                        torch.pi*3/4, torch.pi, -torch.pi*3/4, -torch.pi/2, -torch.pi/4, 0, torch.pi/4, torch.pi/2])

    xyzrpy = torch.tensor([[[0, 0, 0, 0, 0, thetas[0]],
                            [r-r/sqrt2, r/sqrt2, 0, 0, 0, thetas[1]],
                            [r, r, 0, 0, 0, thetas[2]],
                            [r+r/sqrt2, r/sqrt2, 0, 0, 0, thetas[3]],
                            [2*r, 0, 0, 0, 0, thetas[4]],
                            [r+r/sqrt2, -r/sqrt2, 0, 0, 0, thetas[5]],
                            [r, -r, 0, 0, 0, thetas[6]],
                            [r-r/sqrt2, -r/sqrt2, 0, 0, 0, thetas[7]],
                            [0, 0, 0, 0, 0, thetas[8]],
                            [r/sqrt2-r, r/sqrt2, 0, 0, 0, thetas[9]],
                            [-r, r, 0, 0, 0, thetas[10]],
                            [-r/sqrt2-r, r/sqrt2, 0, 0, 0, thetas[11]],
                            [-2*r, 0, 0, 0, 0, thetas[12]],
                            [-r/sqrt2-r, -r/sqrt2, 0, 0, 0, thetas[13]],
                            [-r, -r, 0, 0, 0, thetas[14]],
                            [r/sqrt2-r, -r/sqrt2, 0, 0, 0, thetas[15]],
                            [0, 0, 0, 0, 0, thetas[0]]]], requires_grad=True)

    return getse3(xyzrpy).Exp()

def mpc_configs(dynamics, T):
    n_batch = 1
    n_state, n_ctrl = 7, 6
    Q = torch.tile(torch.eye(n_state + n_ctrl), (n_batch, T, 1, 1))
    Q[..., n_state:, n_state:] *= 1e-1
    p = torch.tile(torch.zeros(n_state + n_ctrl), (n_batch, T, 1))

    stepper = pp.utils.ReduceToBason(steps=1, verbose=False)
    MPC = pp.module.MPC(dynamics, Q, p, T, stepper=stepper)
    return MPC

def main():
    T = 90
    dt = 1
    dynamics = anySE3(dt=dt)
    waypoints = waypoints_configs()
    traj = pp.bspline(waypoints, interval=0.2, extrapolate=True)
    print("traj shape is", traj.shape)
    dynamics.set_reftrajectory(traj)
    MPC = mpc_configs(dynamics, T = T)

    x_init = pp.SE3(torch.tensor([[0, 0, 0, 0, 0, 0, 1]], dtype=torch.float32, requires_grad=True))
    _, u_mpc, _ = MPC(dt, x_init)

    x_traj = waypoints[:,0,:].unsqueeze(-2).repeat((1, T, 1))
    dynamics.recover_dynamics()
    for i in range(T-1):
        x_traj[...,i+1,:], _ = dynamics(x_traj[...,i,:].clone(), u_mpc[...,i,:])

    visualize_traj(x_traj, traj)

if __name__ == "__main__":
    main()
