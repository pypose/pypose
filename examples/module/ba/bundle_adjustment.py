"""
This file is adapted from:
https://github.com/sair-lab/bae/blob/release/ba_example.py
"""

import argparse
import pypose as pp
import torch, torch.nn as nn
from pypose.optim.solver import PCG
from pypose.autograd.function import TT, psjac

from bal_dataset import ba_problem, save_ba


class Reproj(nn.Module):

    def __init__(self, K, C, P):
        # K: intrinsic, C: camera pose, P: point cloud
        super().__init__()
        self.K = pp.Parameter(TT(K))
        self.C = pp.Parameter(TT(C))
        self.P = pp.Parameter(TT(P))

    def forward(self, observe, cidx, pidx):
        return Reproj.project(self.K[cidx], self.C[cidx], self.P[pidx]) - observe

    @psjac
    def project(K, C, P):
        cp = C.Act(P)
        n = - cp[..., :2] / cp[..., [2]]
        radius = n.square().sum(dim=-1, keepdim=True)
        focal, k1, k2 = K[..., :1], K[..., 1:2], K[..., 2:3]
        distortion = 1 + k1 * radius + k2 * radius.square()
        return focal * distortion * n


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Bundle adjustment on a BAL problem")
    datasets = ["ladybug", "trafalgar", "dubrovnik", "venice", "final"]

    parser.add_argument("--dataset", default="trafalgar", choices=datasets)
    parser.add_argument("--problem", default="problem-257-65132-pre")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache-dir", default="./examples/module/ba/data")
    parser.add_argument("--save-dir", default="./examples/module/ba/save")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--reject", type=int, default=30)
    parser.add_argument("--cg-tol", type=float, default=1e-4)
    parser.add_argument("--cg-maxiter", type=int, default=250)
    parser.add_argument("--plot-max-points", type=int, default=8000)
    args = parser.parse_args()

    device = torch.device(args.device)
    assert torch.cuda.is_available(), "Sparse LM currently requires CUDA."
    assert device.type == "cuda", "This sparse BA example must run on CUDA."

    prob = ba_problem(args.problem, args.dataset, args.cache_dir, device)
    inp = {"observe": prob["pixels"], "cidx": prob["cidx"], "pidx": prob["pidx"]}

    model = Reproj(prob["intrinsics"], prob["cameras"], prob["points"]).to(device)

    strategy = pp.optim.strategy.TrustRegion(up=2.0, down=0.5 ** 4)
    solver = PCG(tol=args.cg_tol, maxiter=args.cg_maxiter)
    optimizer = pp.optim.LM(model, solver=solver, strategy=strategy, reject=args.reject, min=1e-6, sparse=True)

    save_ba(model, prob, args.save_dir, args.plot_max_points, "initial", "Initial reconstruction")

    for step in range(args.steps):
        optimizer.step(inp)
        loss = save_ba(model, prob, args.save_dir, args.plot_max_points, f"iter-{step:02d}", f"Iteration {step:02d}")
        print(f"Iteration {step:02d}, loss: {loss:.6f}")

    final = save_ba(model, prob, args.save_dir, args.plot_max_points, "sparse-lm", "Optimized reconstruction")
    print(f"Final mean squared reprojection error: {final:.6f}")
