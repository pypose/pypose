import argparse, os
import torch, pypose as pp
import matplotlib.pyplot as plt
from pypose.lietensor import LieTensor


def angular_vel_2_quaternion_dot(quaternion, w):
    device = quaternion.device
    p, q, r = w

    zero_t = torch.zeros(1, device=device)

    omega_1 = torch.cat((zero_t, -r, q, -p))
    omega_2 = torch.cat((r, zero_t, -p, -q))
    omega_3 = torch.cat((-q, p, zero_t, -r))
    omega_4 = torch.cat((p, q, r, zero_t))

    omega_matrix = torch.stack((omega_1, omega_2, omega_3, omega_4))

    return -0.5 * omega_matrix @ quaternion.T


class MultiCopter(pp.module.NLS):
    def __init__(self, mass, g, J, dt):
        super(MultiCopter, self).__init__()
        self.device = J.device
        self.m = mass
        self.J = J
        self.J_inverse = torch.inverse(self.J)
        self.g = g
        self.tau = dt
        self.e3 = torch.tensor([0., 0., 1.], device=self.device).reshape(3, 1)

    def state_transition(self, state, input, t=None):
        new_state = self.rk4(state, input, self.tau)
        self.pose_normalize(new_state)
        return new_state

    def rk4(self, state, input, t=None):
        k1 = self.xdot(state, input)
        k1_state = state + k1 * t / 2
        self.pose_normalize(k1_state)

        k2 = self.xdot(k1_state, input)
        k2_state = state + k2 * t / 2
        self.pose_normalize(k2_state)

        k3 = self.xdot(k2_state, input)
        k3_state = state + k3 * t
        self.pose_normalize(k3_state)

        k4 = self.xdot(k3_state, input)

        return state + (k1 + 2 * k2 + 2 * k3 + k4) / 6 * t

    def pose_normalize(self, state):
        state[3:7] = state[3:7] / torch.norm(state[3:7])

    def observation(self, state, input, t=None):
        return state

    def xdot(self, state, input):
        pose, vel, angular_speed = state[3:7], state[7:10], state[10:13]
        thrust, M = input[0], input[1:4]

        # convert the 1d row vector to 2d column vector
        M = torch.unsqueeze(M, 0)
        pose = torch.unsqueeze(pose, 0)
        pose_SO3 = LieTensor(pose, ltype=pp.SO3_type)
        Rwb = pose_SO3.matrix()[0]

        acceleration = (Rwb @ (-thrust * self.e3) + self.m * self.g * self.e3) / self.m

        angular_speed = torch.unsqueeze(angular_speed, 1)
        w_dot = self.J_inverse \
            @ (M.T - torch.linalg.cross(angular_speed, self.J @ angular_speed, dim=0))

        # transfer angular_speed from body frame to world frame
        return torch.concat([
                vel,
                torch.squeeze(angular_vel_2_quaternion_dot(pose, angular_speed)),
                torch.squeeze(acceleration),
                torch.squeeze(w_dot)
            ]
        )


def get_sub_states(input_states, sub_state_index):
    sub_states = input_states[..., sub_state_index]
    return sub_states


def subPlot(ax, x, y, xlabel=None, ylabel=None):
    x = x.detach().cpu().numpy()
    y = y.detach().cpu().numpy()
    ax.plot(x, y)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Multicopter Example')
    parser.add_argument("--device", type=str, default='cpu', help="cuda or cpu")
    parser.add_argument("--save", type=str, default='./examples/module/dynamics/save/',
                        help="location of png files to save")
    parser.add_argument('--show', dest='show', action='store_true',
                        help="show plot, default: False")
    parser.set_defaults(show=False)
    args = parser.parse_args(); print(args)
    os.makedirs(os.path.join(args.save), exist_ok=True)

    N = 100    # Number of time steps
    dt = 0.02
    state = torch.zeros(N, 13, device=args.device)
    state[0] = torch.zeros(13, device=args.device)
    state[0][6] = 1
    time  = torch.arange(0, N, device=args.device) * dt
    input = torch.zeros(N, 4, device=args.device)
    input[..., 0] = 2
    input[..., 1:4] = 0.001

    inertia = torch.tensor([[0.0820, 0., 0.00000255],
                            [0., 0.0845, 0.],
                            [0.00000255, 0., 0.1377]], device=args.device)
    model = MultiCopter(0.18,
                    torch.tensor(9.81, device=args.device), inertia, dt).to(args.device)

    # Calculate trajectory
    for i in range(N - 1):
        state[i + 1], _ = model(state[i], input[i])

    # Create time plots to show dynamics
    f, ax = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True)
    ax.set_xlabel("time(s)")
    ax.set_ylabel("Position (m)")
    subPlot(ax, time, get_sub_states(state, 0), ylabel='X')
    subPlot(ax, time, get_sub_states(state, 1), ylabel='Y')
    subPlot(ax, time, get_sub_states(state, 2), ylabel='Z')
    figure = os.path.join(args.save + 'multicoptor.png')
    plt.savefig(figure)
    print("Saved to", figure)

    if args.show:
        plt.show()
