import os
import time
import torch
import argparse
import pypose as pp
import matplotlib.pyplot as plt


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

    N = 100
    dt = 0.1
    state = torch.zeros(N, 1, device=args.device) # positions
    time = torch.arange(0, N, device=args.device) * dt
    ref_state = torch.ones(N-1, 1, device=args.device) # positions at 1m

    pid = pp.module.PID(0.6, 0.1, 0.1)

    for i in range(N - 1):
        error = ref_state[i] - state[i]
        error_dot = (ref_state[i] - state[i]) / dt
        state[i+1] = state[i] + pid.forward(error, error_dot) * dt
        state[i+1] -= 0.05 # apply the stable error

    plt.plot(time.cpu(), state[..., 0].cpu(), '--')
    plt.xlabel('Time (s)')
    plt.ylabel('Position (m)')

    figure = os.path.join(args.save + 'pid_controller.png')
    plt.savefig(figure)
    print("Saved to", figure)

    if args.show:
        plt.show()
