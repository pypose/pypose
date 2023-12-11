import warnings
from typing import TypedDict, Literal

import torch
from .geometry import svdstf
from ..lietensor import mat2SO3, SE3, Sim3, identity_Sim3
from ..lietensor.lietensor import SE3Type, Sim3Type

PoseRelaType = Literal['translation', 'rotation', 'full',
                       'rotation_angle_rad', 'rotation_angle_deg']

RelaType = Literal['translation', 'frame']

class StampedSE3(object):
    def __init__(self, poses_SE3=None, timestamps=None, dtype=torch.float64):
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

    def __getitem__(self, index):
        return StampedSE3(self.poses[index], self.timestamps[index], self.poses.dtype)

    def reduce_to_ids(self, ids) -> None:
        if isinstance(ids, torch.Tensor):
            ids = ids.long().tolist()
        self.timestamps = self.timestamps[ids]
        self.poses = self.poses[ids]

    def align(self, trans):
        if isinstance(trans.ltype, SE3Type):
            self.poses = trans @ self.poses

        elif isinstance(trans.ltype, Sim3Type):
            ones = torch.ones_like(self.poses.data[..., 0:1])
            poses_sim = Sim3(torch.cat((self.poses.data, ones), dim=-1))
            traned_pose = trans @ poses_sim
            self.poses = SE3(traned_pose.data[..., 0:7])

    def translation(self):
        return self.poses.translation()

    def rotation(self):
        return self.poses.rotation()

    def type(self, dtype=torch.float64):
        self.poses = self.poses.to(dtype)

    def cuda(self):
        self.poses = self.poses.cuda()

    def cpu(self):
        self.poses = self.poses.cpu()

    @property
    def num_poses(self):
        return self.poses.shape[0]

    @property
    def first_pose(self):
        return self.poses[0]

    @property
    def dtype(self):
        return self.poses.dtype

    @property
    def device(self):
        return self.poses.device

    @property
    def distance(self) -> torch.tensor:
        trans = self.translation()
        error = torch.linalg.norm(trans[:-1]-trans[1:], dim=-1, dtype=trans.dtype)
        map = torch.zeros_like(trans[:,0])
        map[1:] = error.cumsum_(dim=0)
        return map

def matching_time_indices(stamps_1, stamps_2, max_diff=0.01, offset_2=0.0):
    r"""
    Searches for the best matching timestamps of two lists of timestamps

    Args:
        stamps_1: torch.tensor
            First list of timestamps.
        stamps_2: torch.tensor
            Second list of timestamps
        max_diff: float
            Maximum allowed absolute time difference for associating
        offset_2: float
            The align offset for the second timestamps

    Returns:
        matching_indices_1: list[int].
            indices of timestamps_1 that match with timestamps_1
        matching_indices_2: list[int].
            indices of timestamps_2 that match with timestamps_2
    """
    stamps_2 += offset_2
    diff_mat = (stamps_1[..., None] - stamps_2[None]).abs()
    indices_1 = torch.arange(len(stamps_1), device=stamps_1.device)
    value, indices_2 = diff_mat.min(dim=-1)

    matching_indices_1 = indices_1[value < max_diff].tolist()
    matching_indices_2 = indices_2[value < max_diff].tolist()

    return matching_indices_1, matching_indices_2

def assoc_traj(traj_est, traj_ref, max_diff=0.01, offset_2=0.0, threshold=0.3):
    r"""
    Associates two trajectories by matching their timestamps

    Args:
        traj_est: StampedSE3
            The trajectory for estimation.
        traj_ref: StampedSE3
            The trajectory for reference.
        max_diff: float
            Max allowed absolute time difference (s) for associating
        offset_2: float
            The aligned offset (s) for the second timestamps.
        threshold: float
            The threshold (%) for the matching warning

    Returns:
        traj_est_aligned: StampedSE3
            The aligned estimation trajectory
        traj_ref_aligned: StampedSE3
            The aligned reference trajectory

    Warning:
        The matched stamps are under the threshold
    """
    snd_longer = len(traj_ref.timestamps) > len(traj_est.timestamps)
    traj_long = traj_ref if snd_longer else traj_est
    traj_short = traj_est if snd_longer else traj_ref
    max_pairs = len(traj_short.timestamps)

    matching_indices_short, matching_indices_long = matching_time_indices(
        traj_short.timestamps, traj_long.timestamps, max_diff,
        offset_2 if snd_longer else -offset_2)

    num_matches = len(matching_indices_long)
    traj_short = traj_short[matching_indices_short]
    traj_long = traj_long[matching_indices_long]

    traj_est_aligned = traj_short if snd_longer else traj_long
    traj_ref_aligned = traj_long if snd_longer else traj_short

    assert len(matching_indices_short) == len(matching_indices_long), \
        {r"matching_time_indices returned unequal number of indices"}
    assert num_matches != 0, \
        {f"found no matching timestamps between estimation and reference with max time "
            "diff {max_diff} (s) and time offset {offset_2} (s)"}

    if num_matches < threshold * max_pairs:
        warnings.warn("Alert !!!!!!!!!!!!!!!!!!!!!!! \
                       The estimated trajectory has not enough \
                       timestamps within the GT timestamps. \
                       May be not be enough for aligned and not accurate results.",
                      category=Warning, stacklevel=2)

    print(f"Found {num_matches} of maxiumn {max_pairs} possible matching."
          f"timestamps between estimation and reference with max time"
          f"diff: {max_diff} (s) and time offset: {offset_2} (s).")

    return traj_est_aligned, traj_ref_aligned

def process_data(traj_est, traj_ref, pose_type: PoseRelaType = 'translation'):
    r'''
    To get the error of the pose based on the pose type

    Args:
        traj_est: StampedSE3
            The trajectory for estimation.
        traj_ref: StampedSE3
            The trajectory for reference.
        pose_type: PoseRelaType
            The type of the pose error

    Returns:
        error: The all error of the pose

    Error:
        ValueError: The pose_type is not supported

    Remarks:
        'translation': || t_{est} - t_{ref} ||_2
        'rotation': || R_{est} - R_{ref} ||_2
        'full':  || T_{est} - T_{ref} ||_2
        'rotation_angle_rad': ||Log(R_{est} - R_{ref})||_2
        'rotation_angle_deg': Degree(||Log(R_{est} - R_{ref})||_2)

    '''
    if pose_type == 'translation':
        E = traj_est.translation() - traj_ref.translation()
    else:
        E = (traj_est.poses.Inv() @ traj_ref.poses).matrix()
    if pose_type == 'translation':
        return torch.linalg.norm(E, dim=-1)
    elif pose_type == 'rotation':
        I = torch.eye(3, device=E.device, dtype=E.dtype).expand_as(E[:,:3,:3])
        return torch.linalg.norm((E[:,:3,:3] - I), dim=(-2, -1))
    elif pose_type == 'full':
        I = torch.eye(4, device=E.device, dtype=E.dtype).expand_as(E)
        return torch.linalg.norm((E - I), dim=(-2, -1))
    elif pose_type == 'rotation_angle_rad':
        return mat2SO3(E[:,:3,:3]).euler().norm(dim=-1)
    elif pose_type == 'rotation_angle_deg':
        error = (mat2SO3(E[:,:3,:3]).euler()).norm(dim=-1)
        return torch.rad2deg(error)
    else:
        raise ValueError(f"Unknown pose type: {pose_type}")

def get_result(error) -> dict:
    '''
    statistical data of the error
    '''
    result_dict ={}
    result_dict['max']    = torch.max(error.abs()).item()
    result_dict['min']    = torch.min(error.abs()).item()
    result_dict['mean']   = torch.mean(error.abs()).item()
    result_dict['median'] = torch.median(error.abs()).item()
    result_dict['std']    = torch.std(error.abs()).item()
    result_dict['rmse']   = torch.sqrt(torch.mean(torch.pow(error, 2))).item()
    result_dict['sse']    = torch.sum(torch.pow(error, 2)).item()

    return result_dict

def compute_APE(traj_est, traj_ref, pose_relation: PoseRelaType='translation',
                max_diff=0.01, offset_2=0.0, align=False, correct_scale=False,
                n_to_align=False, align_origin=False, match_thresh = 0.3):
    r'''
    Computes the Absolute Pose Error (RPE) between two trajectories

    Args:
        traj_est: StampedSE3
            The estimated trajectory
        traj_ref: StampedSE3
            The reference trajectory
        pose_relation: PoseRelaType
            The type of the pose error.
            Including: 'translation', 'rotation', 'full',
                        'rotation_angle_rad', 'rotation_angle_deg'.
            The details are in the process_data function.
        max_diff: float
            Max allowed absolute time difference (s) for associating
        offset_2: float
            The aligned offset (s) for the second timestamps.
        align: bool
            If True, the trajectory will be aligned by the scaled svd method.
        correct_scale: bool
            If True, the scale will be corrected by the svd method.
        n_to_align: int
            The number of the trajectory to align.
            If n_to_align == -1, all trajectory will be aligned.
        align_origin: bool
            If True, the trajectory will be aligned by the first pose.
        match_thresh: float
            The threshold for the matching pair.
            i.e. If the matching pairs are under the threshold,
                 the warning will given.
    Returns:
        error: The statics error of the trajectory
    '''

    traj_est, traj_ref = assoc_traj(traj_est, traj_ref, max_diff, offset_2, match_thresh)
    trans_mat = identity_Sim3(1, dtype=traj_est.dtype, device=traj_est.device)

    if align:
        n_to_align = traj_est.num_poses if n_to_align == -1 else n_to_align
        est_trans = traj_est.translation()[:,:n_to_align]
        ref_trans = traj_ref.translation()[:,:n_to_align]
        trans_mat = svdstf(est_trans, ref_trans, correct_scale)

    if align_origin:
        trans_mat[..., :7] = (traj_ref.first_pose @ traj_est.first_pose.Inv()).data

    traj_est.align(trans_mat)

    error = process_data(traj_est, traj_ref, pose_relation)
    result = get_result(error)

    return result

def compute_RPE(traj_est, traj_ref, pose_relation: PoseRelaType='translation',
                max_diff=0.01, offset_2=0.0, align=False, correct_scale=False,
                n_to_align=-1, align_origin=False,
                relation_type: RelaType='frame',
                delta=1.0, tol=0.1, all_pairs=False, match_thresh=0.3):

    r'''
    Computes the Relative Pose Error (RPE) between two trajectories

    Args:
        traj_est: StampedSE3
            The estimated trajectory
        traj_ref: StampedSE3
            The reference trajectory
        pose_relation: PoseRelaType
            The type of the pose error.
            Including: 'translation', 'rotation', 'full',
                        'rotation_angle_rad', 'rotation_angle_deg'.
            The details are in the process_data function.
        max_diff: float
            Max allowed absolute time difference (s) for associating
        offset_2: float
            The aligned offset (s) for the second timestamps.
        align: bool
            If True, the trajectory will be aligned by the scaled svd method.
        correct_scale: bool
            If True, the scale will be corrected by the svd method.
        n_to_align: int
            The number of the trajectory to align.
            If n_to_align == -1, all trajectory will be aligned.
        align_origin: bool
            If True, the trajectory will be aligned by the first pose.
        relation_type: RelaType
            The type of the relation between the two trajectory.
            Including: 'translation', 'frame'
        delta: float
            The delta to select the pair.
        all_pairs: bool
            If True, all pairs will be used for evaluation.
        match_thresh: float
            The threshold for the matching pair.
            i.e. If the matching pairs are under the threshold,
                 the warning will given.
        est_name:
            The estimated trajectory name
        ref_name:
            The reference trajectory name

    Returns: dict
        error: The statics error of the trajectory
    '''
    # Function for get the frame pairs
    def get_frame_pairs(traj, delta, all_pairs):
        traj_len = traj.num_poses
        delta = int(delta)
        assert delta >= 1, "delta must >= 1"
        if all_pairs:
            ids_1 = torch.arange(traj_len, device=traj.device, dtype=torch.long)
            ids_2 = ids_1 + delta
            id_pairs = (ids_1[ids_2<traj_len].tolist(),
                        ids_2[ids_2<traj_len].tolist())
        else:
            ids = torch.arange(0, traj_len, delta, device=traj.device, dtype=torch.long)
            id_pairs = (ids[:-1].tolist(), ids[1:].tolist())

        return id_pairs

    # Function for get the distance pairs
    def get_distance_pairs(traj, delta, all_pairs, tol=0.1):
        acc_dis = traj.distance
        if all_pairs:
            distance_mat = -acc_dis[:, None] + acc_dis[None, :]
            mask = torch.tril(torch.ones_like(distance_mat), diagonal=0)
            distance_mat[mask == 1] = float('inf')
            value, index = torch.min(torch.abs(distance_mat - delta), dim=-1)
            id0 = torch.arange(value.shape[0], device=value.device)
            value_mask = value < tol
            idx_0, idx_1 = id0[value_mask], index[value_mask]
            id_pairs = (idx_0.tolist(), idx_1.tolist())

        else:
            idx = [0]
            source_idx = 0
            for target_idx in range(acc_dis.shape[0]):
                delta_path = acc_dis[target_idx] - acc_dis[source_idx]
                if delta_path >= delta:
                    idx.append(target_idx)
                    source_idx = target_idx
            id_pairs = (idx[:-1], idx[1:])
        return id_pairs

    traj_est, traj_ref = assoc_traj(traj_est, traj_ref, max_diff, offset_2, match_thresh)
    trans_mat = identity_Sim3(1, dtype=traj_est.dtype, device=traj_est.device)

    if align:
        n_to_align = traj_est.num_poses if n_to_align == -1 else n_to_align
        est_trans = traj_est.translation()[:,:n_to_align]
        ref_trans = traj_ref.translation()[:,:n_to_align]
        trans_mat = svdstf(est_trans, ref_trans, correct_scale)

    if align_origin:
        trans_mat[...,:7] = (traj_ref.first_pose @ traj_est.first_pose.Inv()).data

    traj_est.align(trans_mat)

    if relation_type == 'frame':
        sour_id, tar_id = get_frame_pairs(traj_ref, int(delta), all_pairs)
    elif relation_type == 'translation':
        sour_id, tar_id = get_distance_pairs(traj_ref, delta, all_pairs, tol)

    traj_est_rela = traj_est.poses.Inv()[sour_id] @ traj_est[tar_id].poses
    traj_ref_rela = traj_ref.poses.Inv()[sour_id] @ traj_ref[tar_id].poses

    error = process_data(traj_est_rela, traj_ref_rela, pose_relation)
    result = get_result(error)

    return result
