"""
This file is adapted from:
https://github.com/sair-lab/bae/blob/release/ba_example.py
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn

import pypose as pp
from pypose.autograd.function import Track, parallel_for_sparse_jacobian
from pypose.optim.solver import PCG

from ba_visualization import save_bundle_adjustment_visualization
from bal_dataset import get_problem


TARGET_DATASET = "trafalgar"
TARGET_PROBLEM = "problem-257-65132-pre"
OPTIMIZE_INTRINSICS = True
NUM_STEPS = 20
CG_TOL = 1e-4
CG_MAXITER = 250
REJECT_STEPS = 30


@parallel_for_sparse_jacobian
def project(points, camera_pose, intrinsics):
    camera_points = camera_pose.Act(points)
    normalized = -camera_points[..., :2] / camera_points[..., [2]]
    radius_sq = normalized.square().sum(dim=-1, keepdim=True)

    focal = intrinsics[..., :1]
    k1 = intrinsics[..., 1:2]
    k2 = intrinsics[..., 2:3]
    distortion = 1 + k1 * radius_sq + k2 * radius_sq.square()
    return focal * distortion * normalized


class Residual(nn.Module):

    def __init__(self, camera_pose, camera_intrinsics, points):
        super().__init__()
        self.pose = nn.Parameter(Track(camera_pose))
        if OPTIMIZE_INTRINSICS:
            self.intrinsics = nn.Parameter(Track(camera_intrinsics))
        else:
            self.register_buffer("intrinsics", camera_intrinsics)
        self.points = nn.Parameter(Track(points))

    def forward(self, observes, cidx, pidx):
        return project(self.points[pidx], self.pose[cidx], self.intrinsics[cidx]) - observes

def main():
    parser = argparse.ArgumentParser(description="Bundle adjustment on a BAL problem")
    parser.add_argument(
        "--dataset",
        default=TARGET_DATASET,
        choices=["ladybug", "trafalgar", "dubrovnik", "venice", "final"],
    )
    parser.add_argument(
        "--problem",
        default=TARGET_PROBLEM,
        help="BAL problem name, with or without .txt/.bz2",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache-dir", default="./examples/module/ba/data")
    parser.add_argument("--save-dir", default="./examples/module/ba/save")
    parser.add_argument("--plot-max-points", type=int, default=8000)
    args = parser.parse_args()

    device = torch.device(args.device)
    assert torch.cuda.is_available(), "Sparse LM currently requires CUDA."
    assert device.type == "cuda", "This sparse bundle-adjustment example must run on CUDA."

    problem = get_problem(
        problem_name=args.problem,
        dataset=args.dataset,
        cache_dir=args.cache_dir
    )
    print(f"Loaded {problem['problem_name']} from {args.dataset}")

    problem = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in problem.items()
    }
    input = {
        "observes": problem["points_2d"],
        "cidx": problem["camera_index"],
        "pidx": problem["point_index"],
    }

    model = Residual(
        camera_pose=problem["camera_pose"],
        camera_intrinsics=problem["camera_intrinsics"],
        points=problem["points_3d"],
    ).to(device)

    strategy = pp.optim.strategy.TrustRegion(up=2.0, down=0.5 ** 4)
    solver = PCG(tol=CG_TOL, maxiter=CG_MAXITER)
    optimizer = pp.optim.LM(
        model,
        solver=solver,
        strategy=strategy,
        reject=REJECT_STEPS,
        min=1e-6,
        sparse=True,
    )

    save_prefix = f"{args.dataset}-{problem['problem_name']}"
    title_prefix = f"{args.dataset}/{problem['problem_name']}"
    save_bundle_adjustment_visualization(
        model.pose,
        model.intrinsics,
        model.points,
        problem["points_2d"],
        problem["camera_index"],
        problem["point_index"],
        Path(args.save_dir) / f"{save_prefix}-initial.png",
        args.plot_max_points,
        title_prefix,
        "Initial reconstruction",
    )

    for step in range(NUM_STEPS):
        optimizer.step(input)
        loss = save_bundle_adjustment_visualization(
            model.pose,
            model.intrinsics,
            model.points,
            problem["points_2d"],
            problem["camera_index"],
            problem["point_index"],
            Path(args.save_dir) / f"{save_prefix}-iter-{step:02d}.png",
            args.plot_max_points,
            title_prefix,
            f"Iteration {step:02d}",
        )
        print(f"Iteration {step:02d}, loss: {loss:.6f}")

    final_loss = save_bundle_adjustment_visualization(
        model.pose,
        model.intrinsics,
        model.points,
        problem["points_2d"],
        problem["camera_index"],
        problem["point_index"],
        Path(args.save_dir) / f"{save_prefix}-sparse-lm.png",
        args.plot_max_points,
        title_prefix,
        "Optimized reconstruction",
    )
    print(f"Final mean squared reprojection error: {final_loss:.6f}")


if __name__ == "__main__":
    main()
