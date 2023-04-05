import torch, argparse, os
from pypose.module import UKF
from bicycle import Bicycle, bicycle_plot


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='UKF Example')
    parser.add_argument("--device", type=str, default='cpu', help="cuda or cpu")
    parser.add_argument("--k", type=int, default=3, 
                        help='An integer parameter for weighting the sigma points.')
    parser.add_argument("--save", type=str, default='./examples/module/filter/save/', 
                        help="location of png files to save")
    parser.add_argument('--show', dest='show', action='store_true',
                        help="show plot, default: False")
    parser.set_defaults(show=False)
    args = parser.parse_args(); print(args)
    os.makedirs(os.path.join(args.save), exist_ok=True)

    T, N, M = 30, 3, 2  # steps, state dim, input dim
    q, r, p = 0.2, 0.2, 5  # covariance of transition noise, observation noise, and estimation

    input = torch.randn(T, M, device=args.device) * 0.1 + \
            torch.tensor([1, 0], device=args.device)
    state = torch.zeros(T, N, device=args.device)  # true states
    est = torch.randn(T, N, device=args.device) * p  # estimation
    obs = torch.zeros(T, N, device=args.device)  # observation
    P = torch.eye(N, device=args.device).repeat( T, 1, 1) * p ** 2  # estimation covariance
    Q = torch.eye(N, device=args.device) * q ** 2  # covariance of transition
    R = torch.eye(N, device=args.device) * r ** 2  # covariance of observation

    bicycle = Bicycle()
    filter = UKF(bicycle, Q, R).to(args.device)

    for i in range(T - 1):
        w = q * torch.randn(N, device=args.device)
        v = r * torch.randn(N, device=args.device)
        state[i + 1], obs[i] = bicycle(state[i] + w, input[i])  # model measurement
        est[i + 1], P[i + 1] = filter(est[i], obs[i] + v, input[i], P[i], k=args.k)

    bicycle_plot('UKF', state, est, P, save=args.save, show=args.show)
