"""Compare EKF, UKF and PF on the same independent bicycle trajectories."""
import argparse
import os

import matplotlib.pyplot as plt
import torch
import pypose as pp

from bicycle import Bicycle, bicycle_plot


def run(batch=3, steps=30, particles=1000, device='cpu', shared_covariance=False,
        save='./examples/module/filter/save/', show=False):
    torch.manual_seed(435)
    dtype = torch.float64
    n, l = 3, 2
    model = Bicycle().to(device)
    # B is the number of independent systems; T is time. The sigma-point and
    # particle axes are internal to UKF/PF and are not part of these arrays.
    states = torch.zeros(batch, steps, n, dtype=dtype, device=device)
    states[:, 0, 1] = torch.arange(batch, dtype=dtype, device=device) * 2
    inputs = torch.randn(batch, steps, l, dtype=dtype, device=device) * 0.05
    inputs[..., 0] += 0.5
    observations = torch.zeros_like(states)

    scale = torch.ones((), dtype=dtype, device=device) if shared_covariance else \
        torch.linspace(1., 1.5, batch, dtype=dtype, device=device)
    eye = torch.eye(n, dtype=dtype, device=device)
    Q = scale[..., None, None] * eye * 0.02 ** 2
    R = scale[..., None, None] * eye * 0.1 ** 2
    P0 = eye * 0.15 ** 2
    process = torch.distributions.MultivariateNormal(torch.zeros_like(states[:, 0]), Q)
    measurement = torch.distributions.MultivariateNormal(torch.zeros_like(states[:, 0]), R)
    for i in range(steps - 1):
        states[:, i + 1] = model.state_transition(states[:, i], inputs[:, i]) + process.sample()
        observations[:, i + 1] = model.observation(states[:, i + 1], inputs[:, i]) + measurement.sample()

    initial = states[:, 0] + torch.randn(batch, n, dtype=dtype, device=device) * 0.15
    filters = {'EKF': pp.module.EKF(Bicycle()), 'UKF': pp.module.UKF(Bicycle()),
               'PF': pp.module.PF(Bicycle(), particles=particles)}
    if save is not None:
        os.makedirs(save, exist_ok=True)
    for name, filter in filters.items():
        filter = filter.to(device)
        estimates = torch.zeros_like(states)
        covariances = P0.expand(batch, steps, n, n).clone()
        estimates[:, 0] = initial
        for i in range(steps - 1):
            estimates[:, i + 1], covariances[:, i + 1] = filter(
                estimates[:, i], observations[:, i + 1], inputs[:, i], covariances[:, i], Q, R)
        assert torch.isfinite(estimates).all() and torch.isfinite(covariances).all()
        position_error = (estimates[..., :2] - states[..., :2]).square().sum(-1).mean(-1).sqrt()
        heading_delta = estimates[..., 2] - states[..., 2]
        heading_error = torch.atan2(heading_delta.sin(), heading_delta.cos()).square().mean(-1).sqrt()
        print(f'{name}: position RMSE {position_error.tolist()}, heading RMSE (rad) {heading_error.tolist()}')
        for b in range(batch):
            # Plot each trajectory separately, with its positional covariance.
            bicycle_plot(f'{name}_batch{b}', states[b].detach(), estimates[b].detach(),
                         covariances[b, :, :2, :2].detach(), save=save, show=show)
            plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch', type=int, default=3)
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--particles', type=int, default=1000)
    parser.add_argument('--shared-covariance', action='store_true')
    parser.add_argument('--save', default='./examples/module/filter/save/')
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()
    if args.batch < 1 or args.steps < 2 or args.particles < 1:
        parser.error('batch and particles must be positive; steps must be at least two')
    run(**vars(args))
