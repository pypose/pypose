import warnings
from typing import Optional, Union, \
                   List, Tuple

import torch
from .geometry import svdstf
from ..lietensor import mat2SO3
from ..lietensor import SE3, Sim3

class StampedSE3(object):
    """
    a representation with temporal information
    and poses
    """
    def __init__(self,
                poses_SE3: SE3 = None,
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

    def __getitem__(self, index) -> 'StampedSE3':
        return StampedSE3(self.timestamps[index], self.pose[index], self.poses.dtype)

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

    def align(self, trans: Union[SE3, Sim3]):
        if isinstance(trans, SE3):
            self.poses = trans @ self.poses

        elif isinstance(trans, Sim3):
            ones = torch.ones_like(self.pose.data[..., 0:1])
            poses_sim = Sim3(torch.cat((self.poses, ones), dim=-1))
            traned_pose = trans @ poses_sim
            self.poses = SE3(traned_pose.data[..., 0:7])

    def translation(self) -> torch.Tensor:
        return self.poses.translation()

    def rotation(self) -> torch.Tensor:
        return self.poses.rotation()

    def Inv(self) -> 'StampedSE3':
        return StampedSE3(self.poses.Inv())

    def type(self, dtype=torch.float64) -> None:
        self.poses = self.poses.to(dtype)

    def cuda(self) -> None:
        self.poses = self.poses.cuda()

    def cpu(self) -> None:
        self.poses = self.poses.cpu()

    @property
    def num_poses(self) -> int:
        return self.poses.shape[0]

    @property
    def first_pose(self) -> SE3:
        return self.poses[0]

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
        traj_1: StampedSE3, traj_2: StampedSE3,
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
        warnings.warn("Alert !!!!!!!!!!!!!!!!!!!!!!! \
                       The estimated trajectory has not enough \
                       timestamps within the GT timestamps. \
                       May be not be enough for aligned and not accurate results.",
                      category=Warning, stacklevel=2)


    print("Found {} of max. {} possible matching timestamps between \n"
        "{} and {} with max time diff.: {} (s) "
        "and time offset: {} (s).".format(num_matches, max_pairs, first_name,
                                          snd_name, max_diff, offset_2))

    return traj_1, traj_2

def process_data(self, traj_est: SE3, traj_ref: SE3,
                pose_type: str = 'translation') -> None:
        # As evo
        if pose_type == 'translation':
            E = traj_est.translation() - traj_ref.translation()
        else:
            E = (traj_est.Inv() @ traj_ref.poses).matrix()

        print("Compared {} absolute pose pairs.".format(self.E.shape[0]))
        print("Calculating error for {} pose relation...".format((pose_type)))

        if pose_type == 'translation':
            return torch.linalg.norm(E, dim=-1)
        elif pose_type == 'rotation':
            return torch.linalg.norm((E[:,:3,:3] -
                   torch.eye(3, device=E.device,
                   dtype=E.dtype).expand_as(E[:,:3,:3])), dim=(-2, -1))

        elif pose_type == 'full':
            return torch.linalg.norm((E - torch.eye(4, device=E.device,
                    dtype=E.dtype).expand_as(E)), dim=(-2, -1))

        # Rodrigues formula
        elif pose_type == 'rotation_angle_rad':
            return torch.acos((torch.diagonal(E[:,:3,:3], offset=0,
                                dim1=-2, dim2=-1).sum(dim=-1) - 1) / 2).abs()

        elif pose_type == 'rotation_angle_deg':
            error = torch.acos((torch.diagonal(E[:,:3,:3], offset=0,
                                dim1=-2, dim2=-1).sum(dim=-1) - 1) / 2).abs()
            return torch.rad2deg(error)

        # euler angle
        elif pose_type == 'rotation_euler_angle_rad':
            return mat2SO3(E[:,:3,:3]).euler().norm(dim=-1)

        elif pose_type == 'rotation_euler_angle_deg':
            error = (mat2SO3(E[:,:3,:3]).euler()).norm(dim=-1)
            return torch.rad2deg(error)
        else:
            raise ValueError("unsupported pose_relation, \
                             Only support (translation, rotation, full, \
                             rotation_angle_rad, rotation_angle_deg, \
                             rotation_euler_angle_rad, rotation_euler_angle_deg)")

def get_result(error) -> dict:
    result_dict ={}
    result_dict['max']    = torch.max(error.abs()).item()
    result_dict['min']    = torch.min(error.abs()).item()
    result_dict['mean']   = torch.mean(error.abs()).item()
    result_dict['median'] = torch.median(error.abs()).item()
    result_dict['std']    = torch.std(error.abs()).item()
    result_dict['rmse']   = torch.sqrt(torch.mean(torch.pow(error, 2))).item()
    result_dict['sse']    = torch.sum(torch.pow(error, 2)).item()

    return result_dict

def compute_ATE(traj_est: StampedSE3, traj_ref: StampedSE3,
                pose_relation: str = "translation",
                #This part is for sync
                max_diff: float = 0.01, offset_2: float = 0.0,
                #This part is for align
                align: bool = False, correct_scale: bool = False,
                n_to_align: int = -1, align_origin: bool = False,
                ref_name: str = "reference", est_name: str = "estimate"):

    traj_est, traj_ref = associate_trajectories(traj_est, traj_ref,
                            max_diff, offset_2, est_name, ref_name)

    if align:
        if n_to_align == -1:
            trans_mat = svdstf(traj_est.translation(),
                            traj_ref.translation(),
                            correct_scale)
        else:
            trans_mat = svdstf(traj_est.translation()[:n_to_align],
                            traj_ref.translation()[:n_to_align],
                            correct_scale)
    if align_origin:
        trans_mat[...,:7] = (traj_ref.first_pose @ traj_ref.first_pose.Inv()).data

    traj_est.align(trans_mat)

    error = process_data(traj_est, traj_ref, pose_relation)
    result = get_result(error)

    return result
