import torch
import pypose as pp
import matplotlib.pyplot as plt

dt = 1

class Bicycle(pp.module.NLS):
    def __init__(self, dt=None):
        super().__init__()
        self.ref_traj = None
        self.dt = dt

    def origin_state_transition(self, state, input, t=None):
        v, w = input[..., 0:1], input[..., 1:2]
        # zeros = torch.zeros_like(v.repeat(1,4), dtype=torch.float32, requires_grad=True)
        # xyzrpy = torch.cat((v, zeros, w), dim=-1)*self.dt
        zeros = torch.zeros_like(v.repeat(1,3), dtype=torch.float32, requires_grad=True)
        xyzrpy = torch.cat((v*(w*self.dt).cos(), v*(w*self.dt).sin(), zeros, w), dim=-1)*self.dt
        rt = getSE3(xyzrpy)
        # return state*rt
        return (state.Exp()*rt).Log()

    def error_state_transition(self, error_state, input, t=None):
        ref_SE3 = self.ref_traj[...,t,:]
        next_ref_SE3 = self.ref_traj[...,t+1,:]
        # return next_ref_SE3.Inv()*self.origin_state_transition(ref_SE3*error_state, input, t)
        return (next_ref_SE3.Inv()*self.origin_state_transition((ref_SE3*error_state.Exp()).Log(), input, t).Exp()).Log()

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

def visualize_traj_and_input(traj, ref_traj, u_mpc, x_error = None):
    # traj, ref_traj = pp.SE3(traj), pp.SE3(ref_traj)
    traj, ref_traj = pp.se3(traj), pp.SE3(ref_traj)
    x_error = pp.se3(x_error)
    # error = ref_traj.Inv()*traj
    error = ref_traj.Inv()*traj.Exp()
    EuError = (error.translation().norm(dim=-1)).squeeze().detach().numpy()
    AngError = (error.euler()[..., -1]).squeeze().detach().numpy()

    # for i in range(90):
    #     print("i = ", i)
    #     print("The current ref angle is ", ref_traj.euler()[..., i, -1])
    #     print("The next ref angle is ", ref_traj.euler()[..., i+1, -1])
    #     print("The difference angle is ", (ref_traj[...,i+1,:].Inv()*ref_traj[...,i,:]).euler())
    #     print("\n")

    for i in range(90):
        print("i = ", i)
        print("The current angle is ", traj.euler()[..., i, -1])
        print("The reference angle is ", ref_traj.euler()[..., i, -1])
        print("The error angle is ", x_error.euler()[..., i, -1])
        print("--------------------------------------------------")
        print("The current translation is ", traj[...,i,:].translation())
        print("The reference translation is ", ref_traj[...,i,:].translation())
        print("The error translation is ", x_error[...,i,:].translation())
        print("\n")

    x = traj.translation()[...,0].squeeze().detach().numpy()
    y = traj.translation()[...,1].squeeze().detach().numpy()
    z = traj.translation()[...,2].squeeze().detach().numpy()
    x_ref = ref_traj.translation()[...,0].squeeze().detach().numpy()
    y_ref = ref_traj.translation()[...,1].squeeze().detach().numpy()
    z_ref = ref_traj.translation()[...,2].squeeze().detach().numpy()
    u_mpc = u_mpc.squeeze().detach().numpy()

    fig, axs = plt.subplots(3, 1, figsize=(10, 10))

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

    ax2_twin = axs[2].twinx()
    eu_error_line, = axs[2].plot(EuError, label='EuError', color = 'orange')
    ang_error_line, = ax2_twin.plot(AngError, label='AngError', color = 'blue')
    axs[2].set_xlabel('time')
    axs[2].set_ylabel('DisError')
    ax2_twin.set_ylabel('AngError')
    axs[2].set_title('Bicycle error')

    lines = [eu_error_line, ang_error_line]
    labels = [l.get_label() for l in lines]
    axs[2].legend(lines, labels)

    plt.tight_layout()
    plt.show()

def getSE3(xyzrpy):
    xyz = xyzrpy[..., :3]
    rpy = xyzrpy[..., 3:]
    q = pp.euler2SO3(rpy).tensor()
    return pp.SE3(torch.cat([xyz, q], dim=-1))

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

    return getSE3(xyzrpy)

def mpc_configs(dynamics, T):
    n_batch = 1
    n_state, n_ctrl = 6, 2
    # n_state, n_ctrl = 7, 2
    Q = torch.tile(torch.eye(n_state + n_ctrl), (n_batch, T, 1, 1))
    # Q[..., 3:n_state, 3:n_state] *= 100
    Q[..., n_state:, n_state:] *= 0.1
    p = torch.tile(torch.zeros(n_state + n_ctrl), (n_batch, T, 1))

    stepper = pp.utils.ReduceToBason(steps=3, verbose=False)
    MPC = pp.module.MPC(dynamics, Q, p, T, stepper=stepper)
    return MPC

def evaluate_traj(traj, ref_traj):
    traj, ref_traj = pp.SE3(traj), pp.SE3(ref_traj)
    error = ref_traj.Inv()*traj

def test_dynamic():
    dynamics = Bicycle(dt=dt)
    waypoints = waypoints_configs()
    traj = pp.bspline(waypoints, interval=0.2, extrapolate=True)

    x_init = traj[...,0,:].Log()
    # x_rela = pp.SE3(torch.tensor([[0, 0, 0, 0, 0, 0, 1]], dtype=torch.float32, requires_grad=True))
    x_traj = x_init.unsqueeze(-2).repeat((1, 6, 1))
    u = torch.tensor([[1, 0.5]]).repeat((5, 1))
    for i in range(5):
        x_traj[...,i+1,:], _ = dynamics(x_traj[...,i,:].clone(), u[i].unsqueeze(0))

    visualize_traj_and_input(x_traj, traj, u)

def main():
    T = 90
    dynamics = Bicycle(dt=dt)
    MPC = mpc_configs(dynamics, T = T)

    waypoints = waypoints_configs()
    traj = pp.bspline(waypoints, interval=0.2, extrapolate=True)
    dynamics.set_reftrajectory(traj)

    # x_init = traj[...,0,:]
    # x_rela = pp.SE3(torch.tensor([[0, 0, 0, 0, 0, 0, 1]], dtype=torch.float32, requires_grad=True))
    x_init = traj[...,0,:].Log()
    x_rela = pp.SE3(torch.tensor([[0, 0, 0, 0, 0, 0, 1]], dtype=torch.float32, requires_grad=True)).Log()
    x_error, u_mpc, _ = MPC(dt, x_rela)

    x_traj = x_init.unsqueeze(-2).repeat((1, T+1, 1))
    dynamics.recover_dynamics()

    dynamics.systime = torch.tensor(0)
    for i in range(T):
        x_traj[...,i+1,:], _ = dynamics(x_traj[...,i,:].clone(), u_mpc[...,i,:])

    visualize_traj_and_input(x_traj, traj, u_mpc, x_error)

if __name__ == "__main__":
    main()
    # test_dynamic()
