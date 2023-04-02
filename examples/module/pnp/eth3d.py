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

# Download the dataset
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




def decompress_7z(file: str):
    with py7zr.SevenZipFile(os.path.join(file), 'r') as archive:
        archive.extractall(path=root)
    return file

class Decompressor7z(IterDataPipe):
    def __init__(self, dp) -> None:
        self.dp = dp

    def __iter__(self):
        for file in self.dp:
            with py7zr.SevenZipFile(file, 'r') as zip:
                yield from zip.readall().items()  # key: filename 
                    

def ETH3d(root: Union[str, Path]):
    root = os.fspath(root)
    url_dp = IterableWrapper([base_url + archive_name for archive_name in scenes])

    # Cache tar.gz archive
    cache_compressed_dp = url_dp.on_disk_cache(
        filepath_fn=lambda url: os.path.join(root, os.path.basename(url)),
    )
    cache_compressed_dp = HttpReader(cache_compressed_dp).end_caching(same_filepath_fn=True)
    cache_decompressed_dp = cache_compressed_dp.on_disk_cache(
        filepath_fn=lambda tar_path: os.path.join(root, tar_path.split(".")[0])  # for book keeping of the files extracted
    )

    cache_decompressed_dp = Decompressor7z(cache_decompressed_dp).end_caching(
        filepath_fn=lambda file_path: os.path.join(root, file_path)
    )

    list(cache_decompressed_dp)
    return 


if __name__ == '__main__':

    data_cache_directory = 'data_cache_eth3d_dp'
    os.makedirs(data_cache_directory, exist_ok=True)

    ETH3d(data_cache_directory)