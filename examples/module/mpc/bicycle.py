import torch
import pypose as pp
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float32)

class Bicycle(pp.module.NLS):
    def __init__(self, dt=None):
        super().__init__()
        self.ref_traj = None
        self.dt = dt

    def origin_state_transition(self, state, input, t=None):
        v, w = input[..., 0:1], input[..., 1:2]
        zeros = torch.zeros_like(v.repeat(1,4), requires_grad=True)
        xyzrpy = torch.cat((v, zeros, w), dim=-1)*self.dt
        rt = getSE3(xyzrpy)
        return (state.Exp()*rt).Log()

    def error_state_transition(self, error_state, input, t=None):
        v, w = input[..., 0:1], input[..., 1:2]
        zeros = torch.zeros_like(v.repeat(1,4), requires_grad=True)
        xyzrpy = torch.cat((v, zeros, w), dim=-1)*self.dt
        rt = getSE3(xyzrpy)

        ref_SE3 = self.ref_traj[...,t,:]
        next_ref_SE3 = self.ref_traj[...,t+1,:]

        return (next_ref_SE3.Inv()*ref_SE3*error_state.Exp()*rt).Log()

    def state_transition(self, state, input, t):
        if self.ref_traj is None:
            return self.origin_state_transition(state, input, t)
        else:
            return self.error_state_transition(state, input, t)

    def observation(self, state, input, t=None):
        return state

    def set_reftrajectory(self, ref_traj):
        self.ref_traj = ref_traj

    def recover_dynamics(self):
        self.ref_traj = None

def getSE3(xyzrpy):
    xyz = xyzrpy[..., :3]
    rpy = xyzrpy[..., 3:]
    q = pp.euler2SO3(rpy).tensor()
    return pp.SE3(torch.cat([xyz, q], dim=-1))

def visualize_traj_and_input(traj, ref_traj, u_mpc):
    traj, ref_traj = pp.se3(traj), pp.SE3(ref_traj)

    x = traj.translation()[...,0].squeeze().detach().numpy()
    y = traj.translation()[...,1].squeeze().detach().numpy()
    z = traj.translation()[...,2].squeeze().detach().numpy()
    x_ref = ref_traj.translation()[...,0].squeeze().detach().numpy()
    y_ref = ref_traj.translation()[...,1].squeeze().detach().numpy()
    z_ref = ref_traj.translation()[...,2].squeeze().detach().numpy()
    u_mpc = u_mpc.squeeze().detach().numpy()

    fig, axs = plt.subplots(2, 1, figsize=(10, 10))

    axs[0].set_aspect('equal')
    axs[0].plot(x, y, 'o-', label='trajectory', alpha=0.8)
    axs[0].plot(x_ref, y_ref, 'o-', label='reference trajectory', alpha=0.1)
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('y')
    axs[0].set_title('Bicycle trajectory')
    axs[0].legend()

    axs[1].plot(u_mpc[:,0], label='v')
    axs[1].plot(u_mpc[:,1], label='w')
    axs[1].set_xlabel('time')
    axs[1].set_ylabel('input')
    axs[1].set_title('Bicycle input')
    axs[1].legend()

    plt.tight_layout()
    plt.show()

def waypoints_configs():
    r = 3
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

    return getSE3(xyzrpy)

def mpc_configs(dynamics, T):
    n_batch = 1
    n_state, n_ctrl = 6, 2
    Q = torch.tile(torch.eye(n_state + n_ctrl), (n_batch, T, 1, 1))
    Q[..., n_state:, n_state:] *= 1
    p = torch.tile(torch.zeros(n_state + n_ctrl), (n_batch, T, 1))

    stepper = pp.utils.ReduceToBason(steps=30, verbose=False)
    MPC = pp.module.MPC(dynamics, Q, p, T, stepper=stepper)
    return MPC

def traj_analysis(traj, ref_traj):
    T = traj.shape[-2]-1
    traj, ref_traj = pp.se3(traj).Exp(), pp.SE3(ref_traj)
    error_traj = (ref_traj.Inv()*traj).Log()

    flags = error_traj.norm(dim=-1) > 1
    for t in range(T):
        if flags[...,t]:
            print("The error index is: ", t)
            break

    check_idx = 34
    print("the current traj is", traj[...,check_idx,:])
    print("the current ref_traj is", ref_traj[...,check_idx,:])
    print("the current error_traj is", error_traj[...,check_idx,:])
    print("\n")

    print("the next traj is", traj[...,check_idx+1,:])
    print("the next ref_traj is", ref_traj[...,check_idx+1,:])
    print("the next error_traj is", error_traj[...,check_idx+1,:])
    print("\n")

def main():
    dt = 1
    T = 40
    init_idx = 0
    dynamics = Bicycle(dt=dt)
    MPC = mpc_configs(dynamics, T = T)

    waypoints = waypoints_configs()
    traj = pp.bspline(waypoints, interval=0.2, extrapolate=True)
    dynamics.set_reftrajectory(traj[...,init_idx:,:])

    x_init = traj[...,init_idx,:].Log()
    x_rela = (traj[...,init_idx,:].Inv()*x_init.Exp()).Log()
    _, u_mpc, _ = MPC(dt, x_rela)

    x_traj = x_init.unsqueeze(-2).repeat((1, T+1, 1))
    dynamics.recover_dynamics()

    dynamics.systime = torch.tensor(0)
    for i in range(T):
        x_traj[...,i+1,:], _ = dynamics(x_traj[...,i,:].clone(), u_mpc[...,i,:])

    traj_analysis(x_traj, traj[...,init_idx:init_idx+T+1,:])
    visualize_traj_and_input(x_traj, traj, u_mpc)

if __name__ == "__main__":
    main()
