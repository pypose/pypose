from pathlib import Path

import matplotlib.pyplot as plt
import torch


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
def _camera_centers(camera_pose):
    return camera_pose.Inv().translation()


@torch.no_grad()
def _least_square_error(camera_pose, camera_intrinsics, points, cidx, pidx, observes):
    camera_points = camera_pose[cidx].Act(points[pidx])
    normalized = -camera_points[..., :2] / camera_points[..., [2]]
    radius_sq = normalized.square().sum(dim=-1, keepdim=True)

    focal = camera_intrinsics[cidx, :1]
    k1 = camera_intrinsics[cidx, 1:2]
    k2 = camera_intrinsics[cidx, 2:3]
    distortion = 1 + k1 * radius_sq + k2 * radius_sq.square()
    residual = focal * distortion * normalized - observes
    return residual.square().sum(dim=-1).mean()


@torch.no_grad()
def _make_reconstruction_figure(
    points,
    cameras,
    max_points,
    title,
    figure_title=None,
):
    points = _sample_rows(points.detach().cpu(), max_points)
    cameras = cameras.detach().cpu()

    figure = plt.figure(figsize=(6, 6))
    ax = figure.add_subplot(1, 1, 1, projection="3d")
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


@torch.no_grad()
def plot_reconstruction(
    points,
    cameras,
    output_path,
    max_points,
    title="Optimized reconstruction",
    figure_title=None,
):
    figure = _make_reconstruction_figure(
        points,
        cameras,
        max_points=max_points,
        title=title,
        figure_title=figure_title,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    print(f"Saved visualization to {output_path}")


@torch.no_grad()
def save_bundle_adjustment_visualization(
    pose,
    intrinsics,
    points,
    observes,
    cidx,
    pidx,
    output_path,
    max_points,
    title_prefix,
    title,
):
    pose = pose.detach()
    intrinsics = intrinsics.detach()
    points = points.detach()

    loss = _least_square_error(
        pose,
        intrinsics,
        points,
        cidx,
        pidx,
        observes,
    )
    plot_reconstruction(
        points,
        _camera_centers(pose),
        output_path,
        max_points=max_points,
        title=title,
        figure_title=f"{title_prefix} | MSE={loss.item():.6f}",
    )
    return loss.item()
