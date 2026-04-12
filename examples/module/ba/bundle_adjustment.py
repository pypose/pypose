"""
This file is adapted from:
https://github.com/sair-lab/bae/blob/release/ba_example.py
"""

import argparse
import pypose as pp
from pypose.optim import LM
import torch, torch.nn as nn
from pypose.optim.solver import PCG
from pypose.autograd.function import psjac

from bal_dataset import ba_problem, save_ba


class Reproj(nn.Module):

    def __init__(self, K, C, P):
        # K: intrinsic, C: camera pose, P: point cloud
        super().__init__()
        # sjac=True: traceable parameters for sparse Jacobian computation enabled.
        # It is required for the pypose.autograd.function.psjac decorator to work.
        self.K = pp.Parameter(K, sjac=True)
        self.C = pp.Parameter(C, sjac=True)
        self.P = pp.Parameter(P, sjac=True)

    def forward(self, observe, cidx, pidx):
        return Reproj.project(self.K[cidx], self.C[cidx], self.P[pidx]) - observe

    # The decorator psjac parallelizes sparse Jacobian assembly for the batch dimension.
    # psjac is an alias of pypose.autograd.function.parallel_sparse_jacobian.
    @psjac
    def project(K, C, P):
        # See math explanation at https://grail.cs.washington.edu/projects/bal/
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
    parser.add_argument("--save", default="./examples/module/ba/save")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--reject", type=int, default=30)
    parser.add_argument("--cg-tol", type=float, default=1e-4)
    parser.add_argument("--cg-maxiter", type=int, default=250)
    args = parser.parse_args()

    device = torch.device(args.device)
    assert torch.cuda.is_available(), "Sparse LM currently requires CUDA."
    assert device.type == "cuda", "This sparse BA example must run on CUDA."

    prob = ba_problem(args.problem, args.dataset, args.cache_dir, device)
    inp = {"observe": prob["pixels"], "cidx": prob["cidx"], "pidx": prob["pidx"]}

    model = Reproj(prob["intrinsics"], prob["cameras"], prob["points"]).to(device)

    strategy = pp.optim.strategy.TrustRegion(up=2.0, down=0.5 ** 4)
    solver = PCG(tol=args.cg_tol, maxiter=args.cg_maxiter)
    opt = LM(model, solver=solver, strategy=strategy, reject=args.reject, sparse=True)

    save_ba(model, prob, args.save, "initial", "Initial reconstruction")

    for step in range(args.steps):
        opt.step(inp)
        loss = save_ba(model, prob, args.save, f"iter-{step:02d}", f"Iter {step:02d}")
        print(f"Iteration {step:02d}, loss: {loss:.6f}")

    final = save_ba(model, prob, args.save, "sparse-lm", "Optimized reconstruction")
    print(f"Final mean squared reprojection error: {final:.6f}")
