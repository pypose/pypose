import torch, argparse
from pypose.module import PF
from bicycle import Bicycle, bicycle_plot

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PF Example')
    parser.add_argument("--device", type=str, default='cpu', help="cuda or cpu")
    parser.add_argument("--N", type=int, default=3000, help='The number of particle.')
    args = parser.parse_args()

    T, N, M = 30, 3, 2  # steps, state dim, input dim
    q, r, p = 0.2, 0.2, 5  # covariance of transition noise, observation noise, and estimation

    input = torch.randn(T, M, device=args.device) * 0.1 + \
            torch.tensor([1, 0], device=args.device)
    state = torch.zeros(T, N, device=args.device)  # true states
    est = torch.randn(T, N, device=args.device) * p  # estimation
    obs = torch.zeros(T, N, device=args.device)  # observation
    P = torch.eye(N, device=args.device).repeat(T, 1, 1) * p ** 2  # estimation covariance
    Q = torch.eye(N, device=args.device) * q ** 2  # covariance of transition
    R = torch.eye(N, device=args.device) * r ** 2  # covariance of observation
    input = torch.tensor([[ 8.0198e-01,  8.9602e-02],
        [ 1.0161e+00, -4.3740e-03],
        [ 1.1103e+00, -1.5303e-01],
        [ 1.0780e+00,  1.7261e-02],
        [ 9.7867e-01,  9.0914e-02],
        [ 1.0811e+00,  1.0397e-02],
        [ 1.1013e+00, -6.4515e-02],
        [ 7.8016e-01, -1.0904e-01],
        [ 8.1350e-01, -1.1022e-01],
        [ 1.0767e+00,  8.0937e-02],
        [ 9.8763e-01, -1.1447e-01],
        [ 1.0013e+00, -2.2052e-02],
        [ 9.7029e-01, -7.1627e-02],
        [ 1.1042e+00, -6.0520e-02],
        [ 9.8153e-01,  1.3929e-04],
        [ 1.0955e+00,  1.9720e-02],
        [ 1.0777e+00, -3.3230e-02],
        [ 1.1358e+00,  5.3802e-02],
        [ 1.1824e+00,  8.2752e-02],
        [ 1.0677e+00, -4.7641e-02],
        [ 1.0379e+00,  9.1086e-02],
        [ 1.0167e+00,  1.4245e-02],
        [ 8.2271e-01,  1.6676e-02],
        [ 8.2834e-01, -4.8550e-02],
        [ 1.0179e+00, -9.1219e-02],
        [ 1.0076e+00, -1.0713e-01],
        [ 1.0113e+00, -7.0540e-02],
        [ 1.0499e+00, -1.6401e-01],
        [ 1.0807e+00,  1.8299e-02],
        [ 8.3921e-01, -1.5069e-01]])
    est = torch.tensor([[ -5.4184, -14.2866,  -1.1193],
        [  3.5642,  -9.8158,   3.9067],
        [  1.2257,   5.1757,  -4.3157],
        [  1.9977,   4.7019,   5.7055],
        [  6.4311,  11.8620,  -0.4852],
        [  1.7203,  -0.6527,  -2.8271],
        [  2.4101,  -7.6036,  -2.2558],
        [ -3.2450,   5.6927,  -0.2219],
        [  1.2611,  -4.6821,   1.8571],
        [ -1.3086,  -3.7057,   6.2837],
        [ -7.5267,   1.0596,  -1.3340],
        [  2.2993,  -0.9700,  -1.8671],
        [  5.9141,   2.4183,  -8.7077],
        [ -4.7623,  -3.6426,  -6.9631],
        [ -4.4524,  -7.3238,  -2.0131],
        [  2.4418,  -1.8022,   2.3618],
        [  4.2060,  -2.4167,  -7.2566],
        [ -6.2037,  -6.6138,   0.3848],
        [  3.5468,   6.3094,   1.1125],
        [  8.6740,   7.1274,  -1.8484],
        [  2.9318,  -3.1581,  -1.8974],
        [  0.8877,   2.7272,   2.4050],
        [ -0.6206,   1.1510,   4.3225],
        [ -8.0469,  12.2429,  -5.8264],
        [ -5.1141,  -8.2070,   8.2977],
        [ -4.9780,  -5.9629,  -5.8955],
        [ -4.5134,   4.4918,  -6.1600],
        [ -3.6818,  -3.0685,  -5.5050],
        [ -9.0510,   3.0668,  -6.8450],
        [  4.1875,  -4.3662,  -4.0311]])
    bicycle = Bicycle()
    filter = PF(bicycle, Q, R, particle_number=args.N).to(args.device)
    for i in range(T - 1):
        w = q * torch.randn(N, device=args.device)
        v = r * torch.randn(N, device=args.device)
        state[i + 1], obs[i] = bicycle(state[i] + w, input[i])  # model measurement
        est[i + 1], P[i + 1] = filter(est[i], obs[i] + v, input[i], P[i], conv_weight=3)

    bicycle_plot('PF', state, est, P)
