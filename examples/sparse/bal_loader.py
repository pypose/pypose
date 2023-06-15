import os
from functools import partial
from operator import itemgetter, methodcaller
from pathlib import Path
from typing import Union

import numpy as np
from scipy.spatial.transform import Rotation
from torchdata.datapipes.iter import FileOpener, HttpReader, IterableWrapper, \
    IterDataPipe, Zipper, IterKeyZipper, OnlineReader, Decompressor, Concater, \
    MapKeyZipper
from torchvision.transforms import Compose
from bs4 import BeautifulSoup
import pypose as pp
import torch

__ALL__ = ['build_pipeline', 'read_bal_data', 'DATA_URL', 'ALL_DATASETS']

DATA_URL = 'https://grail.cs.washington.edu/projects/bal/'
ALL_DATASETS = ['ladybug', 'trafalgar', 'dubrovnik', 'venice', 'final']

with_base_url = partial(os.path.join, DATA_URL)

def concat(a, b):
    return a + b

def endswith(s, b):
    return s.endswith(b)

def not_none(s):
    return s is not None

def debug(s):
    print(s)
    return s

def problem_lister(*problem_url):
    # extract problem file urls from the problem url
    return OnlineReader(IterableWrapper(problem_url)
    # parse HTML <a> tag's href attributes using bs4
    ).readlines(return_path=False
    ).map(partial(BeautifulSoup, features="html.parser")).map(methodcaller('find', 'a')
    # must end with .bz2
    ).filter(not_none).map(methodcaller('get', 'href')).filter(partial(endswith, b='.bz2')
    # add base url
    ).map(with_base_url)

def download_pipe(cache_dir: Union[str, Path], url_dp, suffix: str):
    cache_compressed = url_dp.on_disk_cache(
        filepath_fn=Compose([os.path.basename, partial(os.path.join, cache_dir)]) ,
    )
    cache_compressed = HttpReader(cache_compressed).end_caching(same_filepath_fn=True)
    # decompress
    cache_decompressed = cache_compressed.on_disk_cache(
        filepath_fn=Compose([partial(str.split, sep=suffix), itemgetter(0)]),
    )
    cache_decompressed = cache_decompressed.open_files(mode="b").load_from_bz2().end_caching(
        same_filepath_fn=True
    )
    return cache_decompressed

def read_bal_data(file_name: str) -> dict:
    """
    Read a Bundle Adjustment in the Large dataset.
    Ref: https://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html

    Parameters
    ----------
    file_name : str
        The decompressed file of the dataset.

    Returns
    -------
    dict
        A dictionary containing the following fields:
        - problem_name: str
            The name of the problem.
        - camera_extrinsics: pp.SE3 (n_cameras, 7)
            The camera extrinsics, represented as pp.SE3.
            First three columns are translation, last four columns is unit quaternion.
        - camera_intrinsics: torch.Tensor (n_cameras, 3, 3)
            The camera intrinsics. Each camera is represented as a 3x3 K matrix.
        - points_3d: torch.Tensor (n_points, 3)
            contains initial estimates of point coordinates in the world frame.
        - points_2d: torch.Tensor (n_observations, 2)
            contains measured 2-D coordinates of points projected on images in each observations.
        - camera_indices: torch.Tensor (n_observations,)
            contains indices of cameras (from 0 to n_cameras - 1) involved in each observation.
        - point_indices: torch.Tensor (n_observations,)
            contains indices of points (from 0 to n_points - 1) involved in each observation.
    """
    with open(file_name, "r") as file:
        n_cameras, n_points, n_observations = map(
            int, file.readline().split())

        camera_indices = np.empty(n_observations, dtype=int)
        point_indices = np.empty(n_observations, dtype=int)
        points_2d = np.empty((n_observations, 2))

        for i in range(n_observations):
            camera_index, point_index, x, y = file.readline().split()
            camera_indices[i] = int(camera_index)
            point_indices[i] = int(point_index)
            points_2d[i] = [float(x), float(y)]

        camera_params = np.empty(n_cameras * 9)
        for i in range(n_cameras * 9):
            camera_params[i] = float(file.readline())
        camera_params = camera_params.reshape((n_cameras, -1))

        points_3d = np.empty(n_points * 3)
        for i in range(n_points * 3):
            points_3d[i] = float(file.readline())
        points_3d = points_3d.reshape((n_points, -1))

    # convert Rodrigues vector to unit quaternion for camera rotation
    # camera_params[0:3] is the Rodrigues vector
    theta = np.linalg.norm(camera_params[:, :3], axis=1)[:, np.newaxis]
    with np.errstate(invalid='ignore'):
        v = camera_params[:, :3] / theta
        v = np.nan_to_num(v)
    r = Rotation.from_rotvec(v)
    q = r.as_quat()

    # use pp.SE3 to represent camera extrinsics as seen in pp.point2pixel
    # camera_params[3:6] is the camera translation
    camera_extrinsics = pp.SE3(np.concatenate([camera_params[:, 3:6], q], axis=1))

    # use torch.Tensor of shape (N, 3, 3) to represent camera intrinsics as seen in pp.point2pixel
    # camera_params[6] is focal length, camera_params[7] and camera_params[8] are two radial distortion parameters
    camera_intrinsics = np.zeros((n_cameras, 3, 3))
    camera_intrinsics[:, 0, 0] = camera_params[:, 6]
    camera_intrinsics[:, 1, 1] = camera_params[:, 6]
    camera_intrinsics[:, 0, 2] = camera_params[:, 7]
    camera_intrinsics[:, 1, 2] = camera_params[:, 8]
    camera_intrinsics[:, 2, 2] = 1
    camera_intrinsics = torch.from_numpy(camera_intrinsics)

    return {'problem_name': os.path.basename(file_name).split('.')[0], # str
            'camera_extrinsics': camera_extrinsics, # pp.SE3 (n_cameras, 7)
            'camera_intrinsics': camera_intrinsics, # torch.Tensor (n_cameras, 3, 3)
            'points_3d': torch.from_numpy(points_3d), # torch.Tensor (n_points, 3)
            'points_2d': torch.from_numpy(points_2d), # torch.Tensor (n_observations, 2)
            'camera_indices': torch.from_numpy(camera_indices), # torch.Tensor (n_observations,)
            'point_indices': torch.from_numpy(point_indices), # torch.Tensor (n_observations,)
            }

def build_pipeline(dataset='ladybug'):
    assert dataset in ALL_DATASETS, f"dataset_name must be one of {ALL_DATASETS}"
    url_dp = problem_lister(with_base_url(dataset + '.html'))
    download_dp = download_pipe(cache_dir='test', url_dp=url_dp, suffix='.bz2')
    bal_data_dp = download_dp.map(read_bal_data)
    return bal_data_dp

if __name__ == '__main__':
    dp = build_pipeline()
    for i in dp:
        print(i)
