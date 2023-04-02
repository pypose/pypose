import os
from pathlib import Path
from typing import Union
import py7zr
import torchdata
from torchdata.datapipes.iter import FileOpener, HttpReader, IterableWrapper, IterDataPipe


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

def decompress7z(file):
    with py7zr.SevenZipFile(file, 'r') as zip:
        return zip.readall().items()  # key: filename relative to the root of the archive, value: byteIO
                    
def download_pipe(root: Union[str, Path]):
    root = os.fspath(root)
    url_dp = IterableWrapper([base_url + archive_name for archive_name in scenes])

    # Cache tar.gz archive
    cache_compressed = url_dp.on_disk_cache(
        filepath_fn=lambda url: os.path.join(root, os.path.basename(url)),
    )
    cache_compressed = HttpReader(cache_compressed).end_caching(same_filepath_fn=True)
    cache_decompressed = cache_compressed.on_disk_cache(
        filepath_fn=lambda tar_path: os.path.join(root, tar_path.split(".")[0])  # for book keeping of the files extracted
    )

    cache_decompressed = cache_decompressed.map(decompress7z).end_caching(
        filepath_fn=lambda file_path: os.path.join(root, file_path)
    )

    return cache_decompressed


def load_pipe(cache_pipe):
    annotation = cache_pipe.filter(lambda x: any(i in x for i in ['cameras.txt', 'images.txt', 'points3D.txt'])).batch(3)
    return annotation


if __name__ == '__main__':

    data_root = 'data_cache_eth3d_dp'
    os.makedirs(data_root, exist_ok=True)

    print(list(load_pipe(download_pipe(data_root))))