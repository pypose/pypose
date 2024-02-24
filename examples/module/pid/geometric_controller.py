import os
import time
import torch
import argparse
import pypose as pp
import matplotlib.pyplot as plt
from torch.linalg import cross
from pypose.module.pid import PID
from pypose.lietensor.basics import vec2skew


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


def skew2vec(input):
    # Convert batched skew matrices to vectors.
    return torch.vstack([-input[1, 2], input[0, 2], -input[0, 1]])


class MultiCopter(pp.module.NLS):
    def __init__(self, mass, g, J, dt):
        super(MultiCopter, self).__init__()
        self.device = J.device
        self.m = mass
        self.J = J
        self.J_inverse = torch.inverse(self.J)
        self.g = g
        self.tau = dt
        self.e3 = torch.tensor([[0., 0., 1.]], device=self.device).reshape(3, 1)

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
        pose_SO3 = pp.LieTensor(pose, ltype=pp.SO3_type)
        Rwb = pose_SO3.matrix()[0]

        acceleration = (Rwb @ (-thrust * self.e3) + self.m * self.g * self.e3) / self.m

        angular_speed = torch.unsqueeze(angular_speed, 1)
        w_dot = self.J_inverse \
            @ (M.T - cross(angular_speed, self.J @ angular_speed, dim=0))

        # transfer angular_speed from body frame to world frame
        return torch.concat([
                vel,
                torch.squeeze(angular_vel_2_quaternion_dot(pose, angular_speed)),
                torch.squeeze(acceleration),
                torch.squeeze(w_dot)
            ]
        )


class GeometricController(torch.nn.Module):
    def __init__(self, parameters, mass, J, g):
        self.device = J.device
        self.parameters = parameters
        self.g = g
        self.m = mass
        self.J = J
        self.e3 = torch.tensor([0., 0., 1.], device=self.device).reshape(3, 1)

    def compute_pose_error(self, pose, ref_pose):
        err_pose =  ref_pose.T @ pose - pose.T @ ref_pose
        return 0.5 * torch.squeeze(skew2vec(err_pose), dim=0)

    def forward(self, state, ref_state):
        device = state.device
        des_pos = torch.unsqueeze(ref_state[0:3], 1)
        des_vel = torch.unsqueeze(ref_state[3:6], 1)
        des_acc = torch.unsqueeze(ref_state[6:9], 1)
        des_acc_dot = torch.unsqueeze(ref_state[9:12], 1)
        des_acc_ddot = torch.unsqueeze(ref_state[12:15], 1)
        des_b1 = torch.unsqueeze(ref_state[15:18], 1)
        des_b1_dot = torch.unsqueeze(ref_state[18:21], 1)
        des_b1_ddot = torch.unsqueeze(ref_state[21:24], 1)

        # extract specific state from state tensor
        position = torch.unsqueeze(state[0:3], 1)
        pose = state[3:7]
        vel = torch.unsqueeze(state[7:10], 1)
        angular_vel = torch.unsqueeze(state[10:13], 1)
        pose_Rwb = pp.LieTensor(pose, ltype=pp.SO3_type).matrix()

        # extract parameters
        kp, kv, kori, kw = self.parameters
        position_pid = PID(kp, 0, kv)
        pose_pid = PID(kori, 0, kw)

        # position controller
        des_b3 = - position_pid.forward(position - des_pos, vel - des_vel) \
            - self.m * self.g * self.e3 \
            + self.m * des_acc

        b3 = pose_Rwb @ self.e3
        thrust_des = torch.squeeze(-des_b3.T @ b3)

        # attitude controller
        err_vel_dot = self.g * self.e3 - thrust_des / self.m * b3 - des_acc
        des_b3_dot = - kp * (vel - des_vel) - kv * err_vel_dot + self.m * des_acc_dot

        # calculate des_b3, des_b3_dot, des_b3_ddot
        b3_dot = pose_Rwb @ vec2skew(torch.squeeze(angular_vel)) @ self.e3
        thrust_dot = torch.squeeze(- des_b3_dot.T @ b3 - des_b3.T @ b3_dot)
        err_vel_ddot = (-thrust_dot * b3 - thrust_des * b3_dot) / self.m - des_acc_dot
        des_b3_ddot = -kp * err_vel_dot - kv * err_vel_ddot + self.m * des_acc_ddot

        des_b3 = -des_b3 / torch.norm(des_b3)
        des_b3_dot = -des_b3_dot / torch.norm(des_b3_dot)
        des_b3_ddot = -des_b3_ddot / torch.norm(des_b3_ddot)

        # calculate des_b2, des_b2_dot, des_b3_ddot
        des_b2 = cross(des_b3, des_b1, dim=0)
        des_b2_dot = cross(des_b3_dot, des_b1, dim=0) + cross(des_b3, des_b1_dot, dim=0)
        des_b2_ddot = cross(des_b3_ddot, des_b1, dim=0) \
            + 2*cross(des_b3_dot, des_b1_dot, dim=0) \
            + cross(des_b3, des_b1_ddot, dim=0)
        des_b2 = des_b2 / torch.norm(des_b2)
        des_b2_dot = des_b2 / torch.norm(des_b2_dot)
        des_b2_ddot = des_b2 / torch.norm(des_b2_ddot)

        # calculate des_b1, des_b1_dot, des_b1_ddot
        des_b1 = cross(des_b2, des_b3, dim=0)
        des_b1_dot = cross(des_b2_dot, des_b3, dim=0) + cross(des_b2, des_b3_dot, dim=0)
        des_b1_ddot = cross(des_b2_ddot, des_b3, dim=0) \
            + 2 * cross(des_b2_dot, des_b3_dot, dim=0) \
            + cross(des_b2, des_b3_ddot, dim=0)
        des_b2 = des_b2 / torch.norm(des_b2)
        des_b1_dot = des_b2 / torch.norm(des_b1_dot)
        des_b1_ddot = des_b2 / torch.norm(des_b1_ddot)

        des_pose_Rwb = torch.concat([des_b1, des_b2, des_b3], dim=1)
        des_pose_Rwb_dot = torch.concat([des_b1_dot, des_b2_dot, des_b3_dot], dim=1)
        des_pose_Rwb_ddot = torch.concat([des_b1_ddot, des_b2_ddot, des_b3_ddot], dim=1)

        des_augular_vel = skew2vec(des_pose_Rwb.T @ des_pose_Rwb_dot)
        wedge_des_augular_vel = vec2skew(des_augular_vel.T)[0]
        des_augular_acc = skew2vec(des_pose_Rwb.T @ des_pose_Rwb_ddot
                                   - wedge_des_augular_vel @ wedge_des_augular_vel)

        M = - pose_pid.forward(self.compute_pose_error(pose_Rwb, des_pose_Rwb),
                               angular_vel - pose_Rwb.T @ (des_pose_Rwb @ des_augular_vel)) \
          + cross(angular_vel, self.J @ angular_vel, dim=0)
        temp_M = torch.squeeze(vec2skew(angular_vel.T)) \
          @ (pose_Rwb.T @ des_pose_Rwb @ des_augular_vel \
          - pose_Rwb.T @ des_pose_Rwb @ des_augular_acc)
        M = (M - self.J @ temp_M).reshape(-1)

        zero_force_tensor = torch.tensor([0.], device=device)
        return torch.concat([torch.max(zero_force_tensor, thrust_des), M])

def get_ref_states(dt, N, device):
    """
    Generate the circle trajectory with fixed heading orientation
    """
    ref_state = torch.zeros(N, 24, device=device)
    time  = torch.arange(0, N, device=args.device) * dt
    # position set points
    ref_state[..., 0:3] = torch.stack(
        [2*(1-torch.cos(time)), 2*torch.sin(time), 0.1*torch.sin(time)], dim=-1)
    # velocity set points
    ref_state[..., 3:6] = torch.stack(
        [2*torch.sin(time), 2*torch.cos(time), 0.1*torch.cos(time)], dim=-1)
    # acceleration set points
    ref_state[..., 6:9] = torch.stack(
        [2*torch.cos(time), -2*torch.sin(time), -0.1*torch.sin(time)], dim=-1)
    # acceleration dot set points
    ref_state[..., 9:12] = torch.stack(
        [-2*torch.sin(time), -2*torch.cos(time), -0.1*torch.cos(time)], dim=-1)
    # acceleration ddot set points
    ref_state[..., 12:15] = torch.stack(
        [-2*torch.cos(time), 2*torch.sin(time), 0.1*torch.sin(time)], dim=-1)
    # b1 axis orientation
    ref_state[..., 15:18] = torch.tensor([[1., 0., 0.]]).view(1, -1)
    # b1 axis orientation dot
    ref_state[..., 18:21] = torch.tensor([[0., 0., 0.]]).view(1, -1)
    # b1 axis orientation ddot
    ref_state[..., 21:24] = torch.tensor([[0., 0., 0.]]).view(1, -1)

    return ref_state

def subPlot(ax, x, y, style, xlabel=None, ylabel=None, label=None):
    x = x.detach().cpu().numpy()
    y = y.detach().cpu().numpy()
    ax.plot(x, y, style, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Geometric controller Example')
    parser.add_argument("--device", type=str, default='cpu', help="cuda or cpu")
    parser.add_argument("--save", type=str, default='./examples/module/pid/save/',
                        help="location of png files to save")
    parser.add_argument('--show', dest='show', action='store_true',
                        help="show plot, default: False")
    parser.set_defaults(show=False)
    args = parser.parse_args(); print(args)
    os.makedirs(os.path.join(args.save), exist_ok=True)

    N = 200    # Number of time steps
    dt = 0.1
    # States: x, y, z, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz
    state = torch.zeros(N, 13, device=args.device)
    state[0][6] = 1
    ref_state = get_ref_states(dt, N, args.device)
    time  = torch.arange(0, N, device=args.device) * dt

    parameters = torch.ones(4, device=args.device) # kp, kv, kori, kw
    mass = torch.tensor(0.18, device=args.device)
    g = torch.tensor(9.81, device=args.device)
    inertia = torch.tensor([[0.0820, 0., 0.00000255],
                            [0., 0.0845, 0.],
                            [0.00000255, 0., 0.1377]], device=args.device)

    controller = GeometricController(parameters, mass, inertia, g)
    model = MultiCopter(mass, g, inertia, dt).to(args.device)

    # Calculate trajectory
    for i in range(N - 1):
        state[i + 1], _ = model(state[i], controller.forward(state[i], ref_state[i]))

    # Create time plots to show dynamics
    f, ax = plt.subplots(nrows=3, sharex=True)
    subPlot(ax[0], time, state[:, 0], '-', ylabel='X position (m)', label='true')
    subPlot(ax[0], time, ref_state[:, 0], '--', ylabel='X position (m)', label='sp')
    subPlot(ax[1], time, state[:, 1], '-', ylabel='Y position (m)', label='true')
    subPlot(ax[1], time, ref_state[:, 1], '--', ylabel='Y position (m)', label='sp')
    subPlot(ax[2], time, state[:, 2], '-', ylabel='Z position (m)', label='true')
    subPlot(ax[2], time, ref_state[:, 2], '--', ylabel='Z position (m)', label='sp')

    ax[0].legend()
    ax[1].legend()
    ax[2].legend()

    figure = os.path.join(args.save + 'geometric_controller.png')
    plt.savefig(figure)
    print("Saved to", figure)

    if args.show:
        plt.show()
