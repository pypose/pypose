import warnings
from typing import Optional, Union, \
                   List, Tuple, \
                   TypedDict, Literal
from dataclasses import dataclass

import torch
from .geometry import svdstf
from ..lietensor import mat2SO3, SE3, Sim3, identity_Sim3
from ..lietensor.lietensor import SE3Type, Sim3Type

PoseRelaType = Literal['translation', 'rotation', 'full',
                       'rotation_angle_rad', 'rotation_angle_deg',
                       'rotation_euler_angle_rad', 'rotation_euler_angle_deg']
class APEConfig(TypedDict):
    pose_relation: PoseRelaType
    max_diff: float
    offset_2: float
    align: bool
    correct_scale: bool
    n_to_align: int
    align_origin: bool

def create_ape_config(pose_relation:PoseRelaType = "translation",
                      max_diff: float = 0.01, offset_2: float = 0.0,
                      align: bool = True, correct_scale:bool = True,
                      n_to_align: int = -1, align_origin: bool = False) -> APEConfig:
    return APEConfig(
        pose_relation = pose_relation, max_diff = max_diff, offset_2 = offset_2,
        align = align, correct_scale = correct_scale, n_to_align = n_to_align,
        align_origin = align_origin)

RelaType = Literal['translation', 'frame']
class RPEConfig(TypedDict):
    pose_relation: PoseRelaType
    relation_type: RelaType
    delta: float
    match_all: bool
    max_diff: float
    offset_2: float
    align: bool
    correct_scale: bool
    n_to_align: int
    align_origin: bool

def create_rpe_config(pose_relation:PoseRelaType = "translation",
                      relation_type: RelaType = "frame", delta: float = 0.0,
                      max_diff: float = 0.01, offset_2: float = 0.0,
                      align: bool = True, correct_scale:bool = True,
                      n_to_align: int = -1, align_origin: bool = False) -> RPEConfig:
    return RPEConfig(
        pose_relation = pose_relation, relation_type = relation_type,
        delta = delta, max_diff = max_diff, offset_2 = offset_2,
        align = align, correct_scale = correct_scale, n_to_align=n_to_align,
        align_origin = align_origin)

class StampedSE3(object):
    def __init__(self,
                poses_SE3: SE3Type = None,
                timestamps: Optional[torch.Tensor] = None,
                dtype: Optional[torch.dtype] = torch.float64):
        r"""
        Class for represent the trajectory with timestamps

        Args:
            poses_SE3: The trajectory poses. Must be SE3
                    e.g. pypose.SE3(torch.rand(10, 7))
            timestamps: The timestamps of the trajectory.
                        Must have same length with poses.
                    e.g torch.tensor(...) or None
            dtype: The data type for poses to calculate (default: torch.float64)
                   The recommended type is torch.float64 or higher

        Returns:
            None

        Error:
            1. ValueError: The poses have shape problem
            2. ValueError: The timestamps have shape problem
        """
        assert (poses_SE3 is not None), {"The pose must be not None"}
        assert (poses_SE3.numel() != 0),  {"The pose must be not empty"}
        assert len(poses_SE3.lshape) == 1, {"Only one trajectory estimation is support,\
                The shape of the trajectory must be 2"}
        self.poses = poses_SE3.to(dtype)

        if timestamps is None:
            print("no timestamps provided, taking index as timestamps")
            self.timestamps = torch.arange(poses_SE3.lshape[0], dtype=torch.float64,
                                           device=poses_SE3.device)
        else:
            self.timestamps = timestamps.type(torch.float64).to(poses_SE3.device)

        assert len(self.timestamps.shape) == 1, {"The timestamp should be one array"}
        assert self.timestamps.shape[0] == self.poses.lshape[0], \
            {"timestamps and poses must have same length"}
        assert torch.all(torch.sort(self.timestamps)[0] == self.timestamps), \
            {"timestamps must be accending"}

    def __getitem__(self, index) -> 'StampedSE3':
        return StampedSE3(self.poses[index], self.timestamps[index],
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

    def align(self, trans: Union[SE3Type, Sim3Type]):
        if isinstance(trans.ltype, SE3Type):
            self.poses = trans @ self.poses

        elif isinstance(trans.ltype, Sim3Type):
            ones = torch.ones_like(self.poses.data[..., 0:1])
            poses_sim = Sim3(torch.cat((self.poses.data, ones), dim=-1))
            traned_pose = trans @ poses_sim
            self.poses = SE3(traned_pose.data[..., 0:7])

    def translation(self) -> torch.Tensor:
        return self.poses.translation()

    def rotation(self) -> torch.Tensor:
        return self.poses.rotation()

    def Inv(self) -> 'StampedSE3':
        return StampedSE3(self.timestamps, self.poses.Inv(),
                          self.poses.type)

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
    def first_pose(self) -> SE3Type:
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
    r"""
    Searches for the best matching timestamps of two lists of timestamps

    Args:
        stamps_1: first list of timestamps
        stamps_2: second list of timestamps
        max_diff: max. allowed absolute time difference for associating
        offset_2: the align offset for the second timestamps

    Returns:
        matching_indices_1: indices of timestamps_1 that match with timestamps_1
        matching_indices_2: indices of timestamps_2 that match with timestamps_2
    """
    stamps_2 += offset_2
    diff_mat = (stamps_1[..., None] - stamps_2[None]).abs()
    indices_1 = torch.arange(len(stamps_1), device=stamps_1.device)
    value, indices_2 = diff_mat.min(dim=-1)

    matching_indices_1 = indices_1[value < max_diff].tolist()
    matching_indices_2 = indices_2[value < max_diff].tolist()

    return matching_indices_1, matching_indices_2

AssocTraj = Tuple[StampedSE3, StampedSE3]
def associate_trajectories(
        traj_1: StampedSE3, traj_2: StampedSE3,
        max_diff: float = 0.01, offset_2: float = 0.0,
        threshold: float = 0.3,
        first_name: str = "first trajectory",
        snd_name: str = "second trajectory") -> AssocTraj:
    r"""
    Associates two trajectories by matching their timestamps

    Args:
        stamps_1:
            first list of timestamps
        stamps_2:
            second list of timestamps
        threshold:
            the threshold of the matching timestamps for aligning
        max_diff:
            max allowed absolute time difference for associating
        offset_2:
            the align offset for the second timestamps
        threshold:
            The threshold for the matching warning
        first_name:
            The name of the first trajectory
        second_name:
            The name of the second trajectory

    Returns:
        matching_indices_1: indices of timestamps_1 that match with timestamps_1
        matching_indices_2: indices of timestamps_2 that match with timestamps_2

    Warning:
        The matched stamps are under the threshold
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

    assert len(matching_indices_short) == len(matching_indices_long), \
        {r"matching_time_indices returned unequal number of indices"}
    assert num_matches != 0, \
        {f"found no matching timestamps between {first_name}"
            "and {snd_name} with max. time "
            "diff {max_diff} (s) and time offset {offset_2} (s)"}

    if num_matches < threshold * max_pairs:
        warnings.warn("Alert !!!!!!!!!!!!!!!!!!!!!!! \
                       The estimated trajectory has not enough \
                       timestamps within the GT timestamps. \
                       May be not be enough for aligned and not accurate results.",
                      category=Warning, stacklevel=2)

    print(f"Found {num_matches} of maxiumn {max_pairs} possible matching."
          f"timestamps between {first_name} and {snd_name} with max time"
          f"diff: {max_diff} (s) and time offset: {offset_2} (s).")

    return traj_1, traj_2

def process_data(traj_est: SE3Type, traj_ref: SE3Type,
                pose_type: PoseRelaType = 'translation') -> None:
    r'''
    To get the error of the pose based on the pose type

    Args:
        traj_est:
            The estimated trajectory
        traj_ref:
            The reference trajectory
        pose_type:
            The type of the pose error

    Returns:
        error: The all error of the pose

    Error:
        ValueError: The pose_type is not supported

        Only PoseRelaType = Literal[ 'translation', 'rotation', 'full',
                'rotation_angle_rad', 'rotation_angle_deg',
                'rotation_euler_angle_rad', 'rotation_euler_angle_deg']
    '''
    if pose_type == 'translation':
        E = traj_est.translation() - traj_ref.translation()
    else:
        E = (traj_est.Inv() @ traj_ref.poses).matrix()

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
        raise ValueError(f"Unknown pose type: {pose_type}")

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

def compute_APE(traj_est: StampedSE3, traj_ref: StampedSE3,
                ape_config: APEConfig, match_thresh: float = 0.3,
                est_name: str = "estimate", ref_name: str = "reference"):
    r'''
    Computes the Absolute Pose Error (RPE) between two trajectories

    Args:
        traj_est:
            The estimated trajectory
        traj_ref:
            The reference trajectory
        match_thresh:
            The threshold for the matching pair.
            i.e. If the matching pairs are under the threshold,
                 the warning will given.
        ape_config:
            The config for APE alignment
        est_name:
            The estimated trajectory name
        ref_name:
            The reference trajectory name

    Returns:
        error: The statics error of the trajectory
    '''
    traj_est, traj_ref = associate_trajectories(traj_est, traj_ref,
                            ape_config['max_diff'], ape_config['offset_2'],
                            match_thresh, est_name, ref_name)

    trans_mat = identity_Sim3(1, dtype=traj_est.dtype,
                              device=traj_est.device)
    if ape_config['align']:
        n_to_align = traj_est.num_poses if ape_config['n_to_align'] == -1 \
                    else ape_config['n_to_align']
        trans_mat = svdstf(traj_est.translation()[:,:n_to_align],
                           traj_ref.translation()[:,:n_to_align],
                           ape_config['correct_scale'])

    if ape_config['align_origin']:
        trans_mat[..., :7] = (traj_ref.first_pose @ traj_est.first_pose.Inv()).data

    traj_est.align(trans_mat)

    error = process_data(traj_est, traj_ref, ape_config['pose_relation'])
    result = get_result(error)

    return result


def compute_RPE(traj_est: StampedSE3, traj_ref: StampedSE3,
                rpe_config: RPEConfig, match_thresh: float = 0.3,
                est_name: str = "estimate", ref_name: str = "reference"):

    r'''
    Computes the Relative Pose Error (RPE) between two trajectories

    Args:
        traj_est:
            The estimated trajectory
        traj_ref:
            The reference trajectory
        match_thresh:
            The threshold for the matching pair.
            i.e. If the matching pairs are under the threshold,
                 the warning will given.
        ape_config:
            The config for APE alignment
        est_name:
            The estimated trajectory name
        ref_name:
            The reference trajectory name

    Returns:
        error: The statics error of the trajectory
    '''
    traj_est, traj_ref = associate_trajectories(traj_est, traj_ref,
                         rpe_config['max_diff'], rpe_config['offset_2'],
                         match_thresh, est_name, ref_name)

    trans_mat = identity_Sim3(1, dtype=traj_est.dtype,
                              device=traj_est.device)
    if rpe_config['align']:
        n_to_align = traj_est.num_poses if rpe_config['n_to_align'] == -1 \
                    else rpe_config['n_to_align']
        trans_mat = svdstf(traj_est.translation()[:,:n_to_align],
                           traj_ref.translation()[:,:n_to_align],
                           rpe_config['correct_scale'])

    if rpe_config['align_origin']:
        trans_mat[...,:7] = (traj_ref.first_pose @ traj_est.first_pose.Inv()).data

    traj_est.align(trans_mat)

    error = process_data(traj_est, traj_ref, rpe_config['pose_relation'])
    result = get_result(error)

    return result
