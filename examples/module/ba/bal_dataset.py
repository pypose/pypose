"""
This file is adapted from:
https://github.com/sair-lab/bae/blob/release/datapipes/bal_loader.py
"""

import bz2
import os
import shutil
import urllib.request
from pathlib import Path

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
        "problem_name": Path(path).stem,
        "camera_params": camera_params,
        "points_3d": points_3d,
        "points_2d": points_2d,
        "camera_index": camera_index,
        "point_index": point_index,
    }


def get_problem(problem_name, dataset, cache_dir="bal_data"):
    _validate_dataset(dataset)
    cache_dir = Path(cache_dir)

    path = _ensure_problem_available(dataset, problem_name, cache_dir)
    problem = read_bal_data(path)
    camera_pose, camera_intrinsics = _split_camera_params(problem["camera_params"])
    problem["camera_pose"] = camera_pose
    problem["camera_intrinsics"] = camera_intrinsics
    return problem
