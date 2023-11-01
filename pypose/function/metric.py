import warnings
from typing import Optional, Union, \
                   List, Tuple

import torch
import pypose

class PoseTrajectory3D(object):
    """
    a representation with temporal information
    and poses
    """
    def __init__(self,
                poses_SE3: pypose.SE3 = None,
                timestamps: Optional[torch.Tensor] = None,
                dtype: Optional[torch.dtype] = torch.float64):
        """
        :param timestamps: optional nx1 list of timestamps give in np.array in seconds
        :param poses_SE3: optional nx7 list of poses (SE3 objects)
        :param dtype: if no special set, the result will be calculated as the pose is
        """
        assert (poses_SE3 is not None), {"The pose must be not empty"}
        assert (poses_SE3.numel() != 0),  {"The pose must be not empty"}
        assert len(poses_SE3.lshape) == 1, {"Only one trajectory estimation is support,\
                The shape of the trajectory must be 2"}

        # change the type of the calculation precise for the pose
        # the recommended type is float64 or higher
        self.poses = poses_SE3.to(dtype)

        if timestamps is None:
            print("no timestamps provided, taking index as timestamps")
            self.timestamps = torch.arange(poses_SE3.lshape[0], dtype=torch.float64,
                                           device=poses_SE3.device)
        else:
            self.timestamps = torch.tensor(timestamps, dtype=torch.float64,
                                           device=poses_SE3.device)

        # check timestamps
        assert len(self.timestamps.shape) == 1, {"The timestamp should be one array"}
        assert self.timestamps.shape[0] == self.poses.lshape[0], \
            {"timestamps and poses must have same length"}
        assert torch.sort(self.timestamps) == self.timestamps, \
            {"timestamps must be accending"}

    def __getitem__(self, index) -> (torch.tensor, pypose.SE3):
        return PoseTrajectory3D(self.timestamps[index], self.pose[index],
                                self.poses.dtype)

    def reduce_to_ids(
            self, ids: Union[List[int], torch.Tensor]) -> None:
        if isinstance(ids, torch.Tensor):
            ids = ids.long().tolist()
        self.timestamps = self.timestamps[ids]
        self.poses = self.poses[ids]

    def reduce_to_time_range(self,
                             start_timestamp: Optional[float] = None,
                             end_timestamp: Optional[float] = None):
        r"""
        Removes elements with timestamps outside of the specified time range.
        :param start_timestamp: any data with lower timestamp is removed
                                if None: current start timestamp
        :param end_timestamp: any data with larger timestamp is removed
                              if None: current end timestamp
        """
        assert self.num_poses, {"trajectory is empty"}
        assert start_timestamp > end_timestamp, \
            {f"start_timestamp is greater than end_timestamp, \
            ({start_timestamp} > {end_timestamp})"}

        if start_timestamp is None:
            start_timestamp = self.timestamps[0]
        if end_timestamp is None:
            end_timestamp = self.timestamps[-1]
        ids = torch.nonzero(self.timestamps >= start_timestamp &\
                            self.timestamps <= end_timestamp)
        self.reduce_to_ids(ids)

    def get_infos(self) -> dict:
        """
        :return: dictionary with some infos about the trajectory
        """
        infos = super(PoseTrajectory3D, self).get_infos()
        infos["duration (s)"] = self.timestamps[-1] - self.timestamps[0]
        infos["t_start (s)"] = self.timestamps[0]
        infos["t_end (s)"] = self.timestamps[-1]
        return infos

    def align(self, trans: Union[pypose.SE3, pypose.Sim3]):
        if isinstance(trans, pypose.SE3):
            self.poses = trans @ self.poses

        elif isinstance(trans, pypose.Sim3):
            ones = torch.ones_like(self.pose.data[:, 0:1])
            poses_sim = pypose.Sim3(torch.cat((self.poses,
                                               ones), dim=-1))
            traned_pose = trans @ poses_sim
            self.poses = pypose.SE3(traned_pose.data[:, 0:7])

    def Inv(self) -> 'PoseTrajectory3D':
        return PoseTrajectory3D(self.poses.Inv())

    def type(self, dtype=torch.float64) -> None:
        self.poses = self.poses.to(dtype)

    def cuda(self) -> None:
        self.poses = self.poses.cuda()

    def cpu(self) -> None:
        self.poses = self.poses.cpu()


    @property
    def translations(self) -> torch.tensor:
        return self.poses.translation()

    @property
    def rotations(self) -> torch.tensor:
        return self.poses.rotation()

    @property
    def dtype(self):
        return self.poses.dtype

    @property
    def device(self):
        return self.poses.device

MatchingIndices = Tuple[List[int], List[int]]
def matching_time_indices(stamps_1: torch.Tensor, stamps_2: torch.Tensor,
                          max_diff: float = 0.01,
                          offset_2: float = 0.0) -> MatchingIndices:
    """
    Searches for the best matching timestamps of two lists of timestamps
    and returns the list indices of the best matches.
    :param stamps_1: short vector of timestamps.
    :param stamps_2: long vector of timestamps.
    :param max_diff: max. allowed absolute time difference
    :param offset_2: optional time offset to be applied to stamps_2
    :return: 2 lists of the matching timestamp indices (stamps_1, stamps_2)
    """
    stamps_2 += offset_2
    diff_mat = (stamps_1[None] - stamps_2[..., None]).abs()
    indices_1 = torch.arange(len(stamps_1), device=stamps_1.device)
    value, indices_2 = diff_mat.min(dim=-1)

    matching_indices_1 = indices_1[value < max_diff].tolist()
    matching_indices_2 = indices_2[value < max_diff].tolist()

    return matching_indices_1, matching_indices_2


def associate_trajectories(
        traj_1: PoseTrajectory3D, traj_2: PoseTrajectory3D,
        max_diff: float = 0.01, offset_2: float = 0.0,
        first_name: str = "first trajectory",
        snd_name: str = "second trajectory"):
    """
    Synchronizes two trajectories by matching their timestamps.
    :param traj_1: trajectory.PoseTrajectory3D object of first trajectory
    :param traj_2: trajectory.PoseTrajectory3D object of second trajectory
    :param max_diff: max. allowed absolute time difference for associating
    :param offset_2: optional time offset of second trajectory
    :param first_name: name of first trajectory for verbose logging
    :param snd_name: name of second trajectory for verbose/debug logging
    :return: traj_1, traj_2 (synchronized)
    """
    if not isinstance(traj_1, PoseTrajectory3D) \
        or not isinstance(traj_2, PoseTrajectory3D):
        raise ValueError("trajectories must be PoseTrajectory3D objects")

    snd_longer = len(traj_2.timestamps) > len(traj_1.timestamps)
    traj_long = traj_2 if snd_longer else traj_1
    traj_short = traj_1 if snd_longer else traj_2
    max_pairs = len(traj_short.timestamps)

    matching_indices_short, matching_indices_long = matching_time_indices(
        traj_short.timestamps, traj_long.timestamps, max_diff,
        offset_2 if snd_longer else -offset_2)

    num_matches = len(matching_indices_long)
    traj_short = traj_short[matching_indices_short]
    traj_long = traj_long[matching_indices_long]

    traj_1 = traj_short if snd_longer else traj_long
    traj_2 = traj_long if snd_longer else traj_short

    assert len(matching_indices_short) != len(matching_indices_long), \
        {r"matching_time_indices returned unequal number of indices"}

    assert num_matches == 0, \
        {f"found no matching timestamps between {first_name}"
            "and {snd_name} with max. time "
            "diff {max_diff} (s) and time offset {offset_2} (s)"}

    if num_matches < 0.3 * max_pairs:
        warnings.warn("Alert !!!!!!!!!!!!!!!!!!!!!!!\
                       The estimated trajectory has not enough \
                       timestamps within the GT timestamps. \
                       May be not be enough for aligned and not accurate results.",
                      category=Warning, stacklevel=2)


    print("Found {} of max. {} possible matching timestamps between \n"
        "{} and {} with max time diff.: {} (s) "
        "and time offset: {} (s).".format(num_matches, max_pairs, first_name,
                                          snd_name, max_diff, offset_2))

    return traj_1, traj_2
