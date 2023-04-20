import os
from functools import partial
from pathlib import Path
from typing import Union
import numpy as np
import py7zr
import pypose as pp
import torchdata
from torchdata.datapipes.iter import FileOpener, HttpReader, IterableWrapper, \
    IterDataPipe, Zipper, IterKeyZipper


full_scenes = [
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


def download_pipe(root: Union[str, Path], scenes=full_scenes):
    root = os.fspath(root)
    # download
    url_dp = IterableWrapper([base_url + archive_name for archive_name in scenes])
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
        filepath_fn=partial(os.path.join, root)
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
    elif '.JPG' in file:
        return 3
    else:
        return None


def load_camera(file):
    # skip first three lines of comments occur in all files
    camera_ids = np.loadtxt(file, skiprows=3, usecols=(0,), dtype=int, ndmin=1) #CAMERA_ID
    camera_data = np.loadtxt(file, skiprows=3, usecols=(2, 3, 4, 5, 6, 7),
                             ndmin=2)  # WIDTH HEIGHT PARAMS[fx, fy, cx, cy,]
    camera_dict = {k.item(): build_intrinsic(*(v[2:])) for k, v in
                   zip(camera_ids, camera_data)}
    return camera_dict


def load_points(file):
    # each row: POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)
    point3d_ids = np.loadtxt(file, usecols=(0,), dtype=int, ndmin=1)  # (N,)
    point3d_xyz = np.loadtxt(file, usecols=(1, 2, 3), dtype=np.float32, ndmin=2)  # (N, 3)
    point3d_rgb = np.loadtxt(file, usecols=(4, 5, 6), dtype=int, ndmin=2)  # (N, 3)

    point3d_dict = {k.item(): (xyz, rgb) for k, xyz, rgb in
                    zip(point3d_ids, point3d_xyz, point3d_rgb)}
    return point3d_dict


def colmap2lietensor(x: np.array):
    return pp.SE3(x[..., [4, 5, 6, 1, 2, 3, 0]])


def parse_image(data):
    ((camera, point, filename), first_line), (_, second_line) = data
    # first line: IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
    pose = np.fromstring(first_line, dtype=np.float32, sep=' ', count=8)[1:]  # (7,)
    first_line = first_line.split()
    image_id = int(first_line[0])
    camera_id = int(first_line[8])
    jpg_name = first_line[9]

    # second line: POINTS2D[] as (X, Y, POINT3D_ID)
    pixels = np.fromstring(second_line, dtype=np.float32, sep=' ').reshape(-1, 3)[:, :2]
    point_ids = np.array([int(i) for i in second_line.split()[2::3]])
    assert len(point_ids) == len(pixels)
    return dict(image_txt=filename, jpg_name=jpg_name, image_id=image_id,
                camera_id=camera_id, pixels=pixels, point_ids=point_ids,
                pose=colmap2lietensor(pose), camera=camera, point=point, )


def load_pipe(cache_pipe):
    camera, point, image, jpg = cache_pipe.demux(4, demux_func, drop_none=True)
    image_file, image_io = FileOpener(image).unzip(2)
    annotation = Zipper(camera.map(load_camera), point.map(load_points), image_file)
    image = Zipper(annotation, image_io)
    image = image.readlines(skip_lines=4).batch(2).map(parse_image)

    return IterKeyZipper(image, jpg,
                        key_fn=lambda x: os.path.basename(x['jpg_name']),
                        ref_key_fn=os.path.basename,
                        merge_fn=lambda x, y: x | {'jpg_path': y},
                        keep_key=False)


if __name__ == '__main__':
    data_root = 'data_cache_eth3d_dp'
    os.makedirs(data_root, exist_ok=True)
    img = load_pipe(download_pipe(data_root))
    print(len(list(img)))
