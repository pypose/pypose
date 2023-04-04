import os
from pathlib import Path
from typing import Union
import numpy as np
import py7zr
import torchdata
from torchdata.datapipes.iter import FileOpener, HttpReader, IterableWrapper, \
    IterDataPipe, Mapper, Zipper

collection = [
    "multi_view_training_dslr_undistorted.7z",
    "multi_view_test_dslr_undistorted.7z",
]

scenes = [
    "courtyard_dslr_undistorted.7z",
    "delivery_area_dslr_undistorted.7z",
    "electro_dslr_undistorted.7z",
    "facade_dslr_undistorted.7z",
    "kicker_dslr_undistorted.7z",
    "meadow_dslr_undistorted.7z",
    "office_dslr_undistorted.7z",
    "pipes_dslr_undistorted.7z",
    "playground_dslr_undistorted.7z",
    "relief_dslr_undistorted.7z",
    "relief_2_dslr_undistorted.7z",
    "terrace_dslr_undistorted.7z",
    "terrains_dslr_undistorted.7z",
    "botanical_garden_dslr_undistorted.7z",
    "boulders_dslr_undistorted.7z",
    "bridge_dslr_undistorted.7z",
    "door_dslr_undistorted.7z",
    "exhibition_hall_dslr_undistorted.7z",
    "lecture_room_dslr_undistorted.7z",
    "living_room_dslr_undistorted.7z",
    "lounge_dslr_undistorted.7z",
    "observatory_dslr_undistorted.7z",
    "old_computer_dslr_undistorted.7z",
    "statue_dslr_undistorted.7z",
    "terrace_2_dslr_undistorted.7z", ]
base_url = 'https://www.eth3d.net/data/'


class Decompressor7z(IterDataPipe):
    def __init__(self, dp) -> None:
        self.dp = dp

    def __iter__(self):
        for file in self.dp:
            with py7zr.SevenZipFile(file, 'r') as zip:
                yield from zip.readall().items()  # key: filename


def download_pipe(root: Union[str, Path]):
    root = os.fspath(root)
    url_dp = IterableWrapper([base_url + archive_name for archive_name in scenes])
    # download
    cache_compressed = url_dp.on_disk_cache(
        filepath_fn=lambda url: os.path.join(root, os.path.basename(url)),
    )
    cache_compressed = HttpReader(cache_compressed).end_caching(same_filepath_fn=True)
    # decompress
    cache_decompressed = cache_compressed.on_disk_cache(
        filepath_fn=lambda tar_path: os.path.join(root, tar_path.split(".")[0])
        # for book keeping of the files extracted
    )
    cache_decompressed = Decompressor7z(cache_decompressed).end_caching(
        filepath_fn=lambda file_path: os.path.join(root, file_path)
    )
    return cache_decompressed


def build_intrinsic(fx, fy, cx, cy):
    return np.array([[fx, 0, cx],
                     [0, fy, cy],
                     [0, 0, 1]])


def demux_func(file):
    if 'cameras.txt' in file:
        return 0
    elif 'points3D.txt' in file:
        return 1
    elif 'images.txt' in file:
        return 2
    else:
        return None


def load_camera(file):
    # skip first three lines of comments occur in all files
    camera_ids = np.loadtxt(file, skiprows=3, usecols=(0,), dtype=int, ndmin=1)  # CAMERA_ID
    camera_data = np.loadtxt(file, skiprows=3, usecols=(2, 3, 4, 5, 6, 7),
                             ndmin=2)  # WIDTH HEIGHT PARAMS[fx, fy, cx, cy,]
    camera_dict = {k.item(): build_intrinsic(*(v[2:])) for k, v in
                   zip(camera_ids, camera_data)}
    return camera_dict


def load_points(file):
    # each row is a point, defined by: POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)
    point3d_ids = np.loadtxt(file, usecols=(0,), dtype=int, ndmin=1)  # (N,)
    point3d_xyz = np.loadtxt(file, usecols=(1, 2, 3), dtype=np.float32, ndmin=2)  # (N, 3)
    point3d_rgb = np.loadtxt(file, usecols=(4, 5, 6), dtype=int, ndmin=2)  # (N, 3)

    point3d_dict = {k.item(): (xyz, rgb) for k, xyz, rgb in
                    zip(point3d_ids, point3d_xyz, point3d_rgb)}
    return point3d_dict


def load_image(data):
    camera, point, (filename, pointer) = data
    image = pointer.readlines()
    image = image[4:]  # discard comments

    for i in range(0, len(image), 2):  # for each image
        # first line: IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
        image_rt = np.fromstring(image[i], dtype=np.float32, sep=' ', count=8)[1:]  # (7,)
        image[i] = image[i].split()
        image_id = int(image[i][0])
        camera_id = int(image[i][8])
        image_name = image[i][9]
        # second line: POINTS2D[] as (X, Y, POINT3D_ID)
        image_points = np.fromstring(image[i + 1], dtype=np.float32, sep=' ').reshape(-1, 3)[:,
                       :2]  # (N, 2)
        point_ids = np.array([int(i) for i in image[i + 1].split()[2::3]])
        assert len(point_ids) == len(image_points)

        # for each image, generate a batch of testing samples
        batched_points2d, batched_points_ids, batched_points_xyz = batches_of_2d_points(
            image_points, point_ids, point3d_xyz_dict, num_points, batch_size)

        # tile rot and t to batch_size
        batched_rot = np.tile(rot[None], (batch_size, 1, 1))
        batched_t = np.tile(t[None], (batch_size, 1))
        batched_camera_intrinsics = np.tile(camera_data_dict[camera_id][None],
                                            (batch_size, 1, 1))


def load_pipe(cache_pipe):
    camera, point, image = cache_pipe.demux(3, demux_func, drop_none=True)

    camera = Mapper(camera, load_camera)
    point = Mapper(point, load_points)
    image = FileOpener(image)

    return Zipper(camera, point, image)

if __name__ == '__main__':

    data_root = 'data_cache_eth3d_dp'
    os.makedirs(data_root, exist_ok=True)

    print(next(iter(load_pipe(download_pipe(data_root)))))