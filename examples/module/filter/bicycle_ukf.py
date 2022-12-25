import torch
import argparse
from pypose.module import UKF
from bicycle import Bicycle

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='UKF Example')
    parser.add_argument("--device", type=str, default='cpu', help="cuda or cpu")
    args = parser.parse_args()
    device = args.device
    T, N, M = 30, 3, 2  # steps, state dim, input dim
    q, r, p = 0.2, 0.2, 5  # covariance of transition noise, observation noise, and estimation
    input = torch.randn(T, M, device=device) * 0.1 + torch.tensor([1, 0], device=device)
    state = torch.zeros(T, N, device=device)  # true states
    est = torch.randn(T, N, device=device) * p  # estimation
    obs = torch.zeros(T, N, device=device)  # observation
    P = torch.eye(N, device=device).repeat(T, 1, 1) * p ** 2  # estimation covariance
    Q = torch.eye(N, device=device) * q ** 2  # covariance of transition
    R = torch.eye(N, device=device) * r ** 2  # covariance of observation

    robot = Bicycle().to(device)
    ukf = UKF(robot, Q, R).to(device)

    for i in range(T - 1):
        w = q * torch.randn(N, device=device)
        v = r * torch.randn(N, device=device)
        state[i + 1], obs[i] = robot(state[i] + w, input[i])  # model measurement
        est[i + 1], P[i + 1] = ukf(est[i], obs[i] + v, input[i], P[i])

    robot.createPlot(state, est, P, 'UKF')
