import os
import time
import torch
import argparse
import pypose as pp
import matplotlib.pyplot as plt
from pypose.module.pid import PID


class DubinCar(pp.module.NLS):
    def __init__(self, dt):
        self.tau = dt
        super(DubinCar, self).__init__()

    # Use RK4 to infer the k+1 state
    def state_transition(self, state, input, t=None):
        return self.rk4(state, input, self.tau)

    def rk4(self, state, input, t=None):
        k1 = self.xdot(state, input)
        k1_state = state + k1 * t / 2

        k2 = self.xdot(k1_state, input)
        k2_state = state + k2 * t / 2

        k3 = self.xdot(k2_state, input)
        k3_state = state + k3 * t

        k4 = self.xdot(k3_state, input)

        return state + (k1 + 2 * k2 + 2 * k3 + k4) / 6 * t

    def observation(self, state, input, t=None):
        return state

    def xdot(self, state, input):
        orientation, vel, w = state[2:]
        # acceleration and angular acceleration
        acc, w_dot = input

        return torch.stack(
            [
                vel * torch.cos(orientation),
                vel * torch.sin(orientation),
                w,
                acc,
                w_dot
            ]
        )


class DubinCarController(torch.nn.Module):
    def __init__(self, parameters):
        super().__init__()
        self.parameters = parameters

    def compute_position_error(self, state, ref_state):
        x_desired, y_desired = ref_state[0:2]
        px, py, ori = state[0:3]
        return torch.cos(ori) * (x_desired - px) + torch.sin(ori) * (y_desired - py)

    def compute_velocity_error(self, state, ref_state):
        v_desired = ref_state[2]
        ori_desired = ref_state[4]

        ori = state[2]
        vel = state[3]

        ori_cos = torch.cos(ori)
        ori_sin = torch.sin(ori)
        des_ori_cos = torch.cos(ori_desired)
        des_ori_sin = torch.sin(ori_desired)

        return (ori_cos * (v_desired * des_ori_cos - vel * ori_cos) \
            + ori_sin * (v_desired * des_ori_sin - vel * ori_sin))

    def forward(self, state, ref_state):
        kp, kv, kori, kw = self.parameters

        v_desired, acc_desired, ori_desired, w_desired, wdot_desired = ref_state[2:]
        orientation = state[2]
        w = state[4]

        ori_cos = torch.cos(orientation)
        ori_sin = torch.sin(orientation)
        des_ori_cos = torch.cos(ori_desired)
        des_ori_sin = torch.sin(ori_desired)

        position_pid = PID(kp, 0, 0)
        velocity_pid = PID(kv, 0, 0)

        # calculate the input of acceleration
        ff = ori_cos * (acc_desired * des_ori_cos - v_desired * w_desired * des_ori_sin) \
            + ori_sin * (acc_desired * des_ori_sin + v_desired * w_desired * des_ori_cos)

        acceleration = \
          position_pid.forward(self.compute_position_error(state, ref_state), 0) \
          + velocity_pid.forward(self.compute_velocity_error(state, ref_state), 0) \
          + ff

        # calculate the input of angular acceleration
        err_angle = ori_desired - orientation
        wdot = kori * err_angle + kw * (w_desired - w) + wdot_desired

        return torch.stack([acceleration, wdot])


def get_ref_states(dt, N, device):
    """
    Generate the trajectory based on constant angular velocity and constant linear velocity
    """
    ref_state = torch.zeros(N, 7, device=device)
    init_state = torch.tensor([0., 0., 0., 1., 1., 0., 0.], device=device)
    ref_state[0] = init_state
    waypoint, ori, desired_vel, des_angular_vel = init_state[0:2], init_state[2], \
        init_state[3], init_state[4]
    zero_tensor = torch.zeros(1, device=device)

    # get the waypoints and orientation at each timestamp using euler intergration
    for i in range(1, N):
        vel = desired_vel * torch.stack([torch.cos(ori), torch.sin(ori)])
        waypoint += vel * dt
        ori += des_angular_vel * dt

        ref_state[i] = torch.tensor(
          [waypoint[0], waypoint[1], desired_vel, zero_tensor, \
            ori, des_angular_vel, zero_tensor], device=device)

    return ref_state


def subPlot(ax, x, y, style, xlabel=None, ylabel=None, label=None):
    x = x.detach().cpu().numpy()
    y = y.detach().cpu().numpy()
    ax.plot(x, y, style, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PID controller Example')
    parser.add_argument("--device", type=str, default='cpu', help="cuda or cpu")
    parser.add_argument("--save", type=str, default='./examples/module/pid/save/',
                        help="location of png files to save")
    parser.add_argument('--show', dest='show', action='store_true',
                        help="show plot, default: False")
    parser.set_defaults(show=False)
    args = parser.parse_args(); print(args)
    os.makedirs(os.path.join(args.save), exist_ok=True)

    N = 100    # Number of time steps
    dt = 0.1
    state = torch.zeros(N, 5, device=args.device)
    ref_state = get_ref_states(dt, N, args.device)
    time  = torch.arange(0, N, device=args.device) * dt

    parameters = torch.ones(4, device=args.device)
    controller = DubinCarController(parameters)

    model = DubinCar(dt).to(args.device)

    # Calculate trajectory
    for i in range(N - 1):
        state[i + 1], _ = model(state[i], controller.forward(state[i], ref_state[i]))

    # Create time plots to show dynamics
    f, ax = plt.subplots(nrows=3, sharex=True)
    subPlot(ax[0], time, state[:, 0], '-', ylabel='X position (m)', label='true')
    subPlot(ax[0], time, ref_state[:, 0], '--', ylabel='X position (m)', label='sp')
    subPlot(ax[1], time, state[:, 1], '-', ylabel='Y position (m)', label='true')
    subPlot(ax[1], time, ref_state[:, 1], '--', ylabel='Y position (m)', label='sp')
    subPlot(ax[2], time, state[:, 2] / torch.pi * 180, '-', ylabel='Orientation (degree)', label='true')
    subPlot(ax[2], time, ref_state[:, 4] / torch.pi * 180, '--', ylabel='Orientation (degree)', label='sp')

    ax[0].legend()
    ax[1].legend()
    ax[2].legend()

    figure = os.path.join(args.save + 'dubincar.png')
    plt.savefig(figure)
    print("Saved to", figure)

    if args.show:
        plt.show()
