import argparse
from io import BytesIO
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from PIL import Image

import pypose as pp

from bal_dataset import get_problem

try:
    from bae.autograd.function import TrackingTensor, map_transform
    from bae.utils.pysolvers import PCG
except ImportError as exc:
    TrackingTensor = None
    _BAE_IMPORT_ERROR = exc

    def map_transform(function):
        return function
else:
    _BAE_IMPORT_ERROR = None


TARGET_DATASET = "trafalgar"
TARGET_PROBLEM = "problem-257-65132-pre"
OPTIMIZE_INTRINSICS = True
NUM_CAMERA_PARAMS = 10 if OPTIMIZE_INTRINSICS else 7
NUM_STEPS = 20
CG_TOL = 1e-4
CG_MAXITER = 250
REJECT_STEPS = 30


def _require_sparse_dependencies(device):
    if TrackingTensor is None:
        raise ImportError(
            "Sparse LM requires bae. Install it with "
            "'pip install git+https://github.com/zitongzhan/bae.git'."
        ) from _BAE_IMPORT_ERROR
    if not torch.cuda.is_available():
        raise RuntimeError("Sparse LM currently requires CUDA, but no CUDA is available.")
    if device.type != "cuda":
        raise RuntimeError("This sparse bundle-adjustment example must run on CUDA.")


@map_transform
def project(points, camera_params):
    camera_points = pp.SE3(camera_params[..., :7]).Act(points)
    normalized = -camera_points[..., :2] / camera_points[..., [2]]
    radius_sq = normalized.square().sum(dim=-1, keepdim=True)

    focal = camera_params[..., [-3]]
    k1 = camera_params[..., [-2]]
    k2 = camera_params[..., [-1]]
    distortion = 1 + k1 * radius_sq + k2 * radius_sq.square()
    return focal * distortion * normalized


class Residual(nn.Module):

    def __init__(self, camera_params, points):
        super().__init__()
        self.pose = nn.Parameter(TrackingTensor(camera_params))
        self.pose.trim_SE3_grad = True
        self.points = nn.Parameter(TrackingTensor(points))

    def forward(self, observes, cidx, pidx):
        return project(self.points[pidx], self.pose[cidx]) - observes


def least_square_error(camera_params, points, cidx, pidx, observes):
    residual = project(points[pidx], camera_params[cidx]) - observes
    return residual.square().sum(dim=-1).mean()


@torch.no_grad()
def camera_centers(camera_params):
    return pp.SE3(camera_params[:, :7]).Inv().translation()


def _sample_rows(points, count):
    if points.shape[0] <= count:
        return points
    index = torch.linspace(0, points.shape[0] - 1, count).long()
    return points[index]


def _focus_points(points):
    points = points.float()
    if points.shape[0] < 32:
        return points

    center = points.median(dim=0).values
    distance = torch.linalg.norm(points - center, dim=-1)
    cutoff = torch.quantile(distance, 0.85)
    inliers = points[distance <= cutoff]
    return inliers if inliers.shape[0] >= 16 else points


def _set_axes_equal(ax, xyz):
    xyz = _focus_points(xyz)
    if xyz.shape[0] >= 16:
        quantiles = torch.tensor([0.05, 0.95], dtype=xyz.dtype, device=xyz.device)
        mins, maxs = torch.quantile(xyz, quantiles, dim=0)
    else:
        mins = xyz.min(dim=0).values
        maxs = xyz.max(dim=0).values
    center = (mins + maxs) * 0.5
    radius = max((maxs - mins).max().item() * 0.7, 1e-3)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


@torch.no_grad()
def _make_reconstruction_figure(
    initial_points,
    current_points,
    initial_cameras,
    current_cameras,
    max_points,
    current_title,
    figure_title=None,
):
    initial_points = _sample_rows(initial_points.detach().cpu(), max_points)
    current_points = _sample_rows(current_points.detach().cpu(), max_points)
    initial_cameras = initial_cameras.detach().cpu()
    current_cameras = current_cameras.detach().cpu()

    figure = plt.figure(figsize=(12, 6))
    titles = ("Initial reconstruction", current_title)
    point_sets = (initial_points, current_points)
    camera_sets = (initial_cameras, current_cameras)

    for axis_id, (title, points, cameras) in enumerate(
        zip(titles, point_sets, camera_sets), start=1
    ):
        ax = figure.add_subplot(1, 2, axis_id, projection="3d")
        ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            c=points[:, 2],
            cmap="viridis",
            s=2,
            alpha=0.55,
        )
        ax.scatter(cameras[:, 0], cameras[:, 1], cameras[:, 2], c="#d1495b", marker="^", s=40)
        ax.set_title(title)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        _set_axes_equal(ax, points)

    if figure_title is not None:
        figure.suptitle(figure_title)
    return figure


def _render_frame(figure):
    buffer = BytesIO()
    figure.tight_layout()
    figure.savefig(buffer, format="png", dpi=180)
    plt.close(figure)
    buffer.seek(0)
    image = Image.open(buffer).convert("RGB")
    frame = image.copy()
    image.close()
    buffer.close()
    return frame


def save_gif(frames, output_path, duration_ms):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )
    print(f"Saved animation to {output_path}")


@torch.no_grad()
def plot_reconstruction(
    initial_points,
    current_points,
    initial_cameras,
    current_cameras,
    output_path,
    max_points,
    current_title="Optimized reconstruction",
    figure_title=None,
):
    figure = _make_reconstruction_figure(
        initial_points,
        current_points,
        initial_cameras,
        current_cameras,
        max_points=max_points,
        current_title=current_title,
        figure_title=figure_title,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    print(f"Saved visualization to {output_path}")


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
    parser.add_argument("--cache-dir", default="./examples/module/bundle_adjustment/data")
    parser.add_argument("--save-dir", default="./examples/module/bundle_adjustment/save")
    parser.add_argument("--loader-source", choices=("local", "bae"), default="local")
    parser.add_argument("--refresh-remote-loader", action="store_true")
    parser.add_argument("--plot-max-points", type=int, default=8000)
    parser.add_argument("--gif-duration-ms", type=int, default=300)
    args = parser.parse_args()

    device = torch.device(args.device)
    _require_sparse_dependencies(device)

    problem = get_problem(
        problem_name=args.problem,
        dataset=args.dataset,
        cache_dir=args.cache_dir,
        use_quat=True,
        loader_source=args.loader_source,
        refresh_loader=args.refresh_remote_loader,
    )
    print(f"Loaded {problem['problem_name']} from {args.dataset}")

    problem = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in problem.items()
    }
    input = {
        "observes": problem["points_2d"],
        "cidx": problem["camera_index_of_observations"],
        "pidx": problem["point_index_of_observations"],
    }

    model = Residual(
        camera_params=problem["camera_params"][:, :NUM_CAMERA_PARAMS].clone(),
        points=problem["points_3d"].clone(),
    ).to(device)

    initial_cameras = model.pose.tensor().detach().clone()
    initial_points = model.points.tensor().detach().clone()
    initial_loss = least_square_error(
        initial_cameras,
        initial_points,
        problem["camera_index_of_observations"],
        problem["point_index_of_observations"],
        problem["points_2d"],
    )
    print(f"Initial mean squared reprojection error: {initial_loss.item():.6f}")
    initial_camera_centers = camera_centers(initial_cameras)
    frames = [
        _render_frame(
            _make_reconstruction_figure(
                initial_points,
                initial_points,
                initial_camera_centers,
                initial_camera_centers,
                max_points=args.plot_max_points,
                current_title="Iteration 00",
                figure_title=(
                    f"{args.dataset}/{problem['problem_name']} | "
                    f"MSE={initial_loss.item():.6f}"
                ),
            )
        )
    ]

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

    start = perf_counter()
    for step in range(NUM_STEPS):
        optimizer.step(input)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        current_cameras = model.pose.tensor().detach()
        current_points = model.points.tensor().detach()
        current_loss = least_square_error(
            current_cameras,
            current_points,
            problem["camera_index_of_observations"],
            problem["point_index_of_observations"],
            problem["points_2d"],
        )
        print(
            f"Iteration {step:02d}: loss={current_loss.item():.6f}, "
            f"elapsed={perf_counter() - start:.3f}s"
        )
        frames.append(
            _render_frame(
                _make_reconstruction_figure(
                    initial_points,
                    current_points,
                    initial_camera_centers,
                    camera_centers(current_cameras),
                    max_points=args.plot_max_points,
                    current_title=f"Iteration {step + 1:02d}",
                    figure_title=(
                        f"{args.dataset}/{problem['problem_name']} | "
                        f"MSE={current_loss.item():.6f}"
                    ),
                )
            )
        )

    optimized_cameras = model.pose.tensor().detach()
    optimized_points = model.points.tensor().detach()
    optimized_loss = least_square_error(
        optimized_cameras,
        optimized_points,
        problem["camera_index_of_observations"],
        problem["point_index_of_observations"],
        problem["points_2d"],
    )
    print(f"Final mean squared reprojection error: {optimized_loss.item():.6f}")

    figure_name = f"{args.dataset}-{problem['problem_name']}-sparse-lm.png"
    gif_name = f"{args.dataset}-{problem['problem_name']}-sparse-lm.gif"
    save_gif(frames, Path(args.save_dir) / gif_name, duration_ms=args.gif_duration_ms)
    plot_reconstruction(
        initial_points,
        optimized_points,
        initial_camera_centers,
        camera_centers(optimized_cameras),
        Path(args.save_dir) / figure_name,
        max_points=args.plot_max_points,
        figure_title=(
            f"{args.dataset}/{problem['problem_name']} | "
            f"final MSE={optimized_loss.item():.6f}"
        ),
    )


if __name__ == "__main__":
    main()
