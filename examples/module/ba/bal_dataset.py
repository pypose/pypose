"""
This file is adapted from:
https://github.com/sair-lab/bae/blob/release/datapipes/bal_loader.py
"""

import bz2
import os
import shutil
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import pypose as pp

DTYPE = torch.float64
DATA_URL = "https://grail.cs.washington.edu/projects/bal/"
ALL_DATASETS = ("ladybug", "trafalgar", "dubrovnik", "venice", "final")


def _normalize_problem_name(problem_name):
    name = os.path.basename(problem_name)
    for suffix in (".txt.bz2", ".txt", ".bz2"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _validate_dataset(dataset):
    if dataset not in ALL_DATASETS:
        raise ValueError(f"dataset must be one of {ALL_DATASETS}, got {dataset!r}")


def _download_url(url, destination, user_agent="pypose-bal-loader/1.0", timeout=60.0):
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": user_agent})
    tmp = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response, tmp.open("wb") as file:
            shutil.copyfileobj(response, file)
        os.replace(tmp, destination)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return destination


def _problem_paths(cache_dir, problem_name):
    normalized = _normalize_problem_name(problem_name)
    return cache_dir / f"{normalized}.txt", cache_dir / f"{normalized}.txt.bz2"


def _problem_url(dataset, problem_name):
    normalized = _normalize_problem_name(problem_name)
    return f"{DATA_URL}data/{dataset}/{normalized}.txt.bz2"


def _ensure_problem_available(dataset, problem_name, cache_dir):
    txt_path, archive_path = _problem_paths(cache_dir, problem_name)
    if txt_path.exists() and txt_path.stat().st_size > 0:
        return txt_path

    if not archive_path.exists() or archive_path.stat().st_size == 0:
        _download_url(_problem_url(dataset, problem_name), archive_path)

    tmp = txt_path.with_suffix(txt_path.suffix + ".tmp")
    try:
        with bz2.open(archive_path, "rb") as compressed, tmp.open("wb") as decompressed:
            shutil.copyfileobj(compressed, decompressed)
        os.replace(tmp, txt_path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return txt_path


def _rotvec_to_quat_xyzw(rotvec):
    theta = torch.linalg.norm(rotvec, dim=-1, keepdim=True)
    half_theta = theta * 0.5
    sin_half = torch.sin(half_theta)
    cos_half = torch.cos(half_theta)
    scale = torch.where(
        theta > 1e-12,
        sin_half / theta,
        0.5 - theta.square() / 48.0,
    )
    xyz = rotvec * scale
    return torch.cat((xyz, cos_half), dim=-1)


def _split_camera_params(camera_params):
    camera_pose = pp.SE3(camera_params[:, :7].clone())
    camera_intrinsics = camera_params[:, 7:].clone()
    return camera_pose, camera_intrinsics


def read_bal_data(path, use_quat=True):
    with open(path, "r", encoding="utf-8") as file:
        n_cameras, n_points, n_observations = map(int, file.readline().split())

        camera_index = torch.empty(n_observations, dtype=torch.int64)
        point_index = torch.empty(n_observations, dtype=torch.int64)
        points_2d = torch.empty((n_observations, 2), dtype=DTYPE)
        for row in range(n_observations):
            camera_id, point_id, x, y = file.readline().split()
            camera_index[row] = int(camera_id)
            point_index[row] = int(point_id)
            points_2d[row, 0] = float(x)
            points_2d[row, 1] = float(y)

        camera_params = torch.empty((n_cameras, 9), dtype=DTYPE)
        for row in range(n_cameras):
            for col in range(9):
                camera_params[row, col] = float(file.readline())

        points_3d = torch.empty((n_points, 3), dtype=DTYPE)
        for row in range(n_points):
            for col in range(3):
                points_3d[row, col] = float(file.readline())

    if use_quat:
        translation = camera_params[:, 3:6]
        rotation = _rotvec_to_quat_xyzw(camera_params[:, :3])
        camera_params = torch.cat((translation, rotation, camera_params[:, 6:]), dim=-1)
    else:
        camera_params = torch.cat((camera_params[:, 3:6], camera_params[:, :3], camera_params[:, 6:]), dim=-1)

    return {
        "name": Path(path).stem,
        "camera_params": camera_params,
        "points": points_3d,
        "pixels": points_2d,
        "cidx": camera_index,
        "pidx": point_index,
    }


def ba_problem(problem_name, dataset, cache_dir="bal_data", device=None):
    _validate_dataset(dataset)
    cache_dir = Path(cache_dir)

    path = _ensure_problem_available(dataset, problem_name, cache_dir)
    problem = read_bal_data(path)
    camera_pose, intrinsics = _split_camera_params(problem["camera_params"])
    problem["cameras"] = camera_pose
    problem["intrinsics"] = intrinsics
    if device is not None:
        problem = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in problem.items()
        }
    print(f"Loaded {problem['name']} from {dataset}")
    problem["dataset"] = dataset
    return problem


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
def save_ba(
    model,
    problem,
    save_dir,
    suffix,
    title,
    max_points=8000,
):
    prefix = f"{problem['dataset']}-{problem['name']}"
    title_prefix = f"{problem['dataset']}/{problem['name']}"
    output_path = Path(save_dir) / f"{prefix}-{suffix}.png"

    pose = model.C.detach()
    intrinsics = model.K.detach()
    points = model.P.detach()
    observes = problem["pixels"]
    cidx = problem["cidx"]
    pidx = problem["pidx"]

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
