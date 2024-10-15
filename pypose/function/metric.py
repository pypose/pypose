import warnings
from typing import TypedDict, Literal

import torch
from .geometry import svdstf
from ..lietensor import mat2SO3, SE3, Sim3, identity_Sim3
from ..lietensor.lietensor import SE3Type, Sim3Type

PoseRelaType = Literal['translation', 'rotation', 'full',
                       'rotation_angle_rad', 'rotation_angle_deg']

Unit = Literal['frame', 'meters']

MericType = Literal['ape', 'rpe']

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
    def accumulated_distances(self) -> torch.tensor:
        trans = self.translation()
        return torch.cat(
            (torch.zeros(1, dtype=trans.dtype), torch.cumsum(torch.linalg.norm(trans[:-1] - trans[1:], dim=-1, dtype=trans.dtype), dim=0)))

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

def associate_trajectories(traj_ref, traj_est, max_diff=0.01, offset_2=0.0, threshold=0.3):
    r"""
    Associates two trajectories by matching their timestamps
    Args:
        traj_ref: StampedSE3
            The trajectory for reference.
        traj_est: StampedSE3
            The trajectory for estimation.
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
    snd_longer = len(traj_est.timestamps) > len(traj_ref.timestamps)
    traj_long = traj_est if snd_longer else traj_ref
    traj_short = traj_ref if snd_longer else traj_est
    max_pairs = len(traj_short.timestamps)

    matching_indices_short, matching_indices_long = matching_time_indices(
        traj_short.timestamps, traj_long.timestamps, max_diff,
        offset_2 if snd_longer else -offset_2)

    assert len(matching_indices_short) == len(matching_indices_long), \
        {r"matching_time_indices returned unequal number of indices"}

    num_matches = len(matching_indices_long)
    traj_short = traj_short[matching_indices_short]
    traj_long = traj_long[matching_indices_long]

    traj_ref_aligned = traj_short if snd_longer else traj_long
    traj_est_aligned = traj_long if snd_longer else traj_short

    assert num_matches != 0, \
        {f"found no matching timestamps between estimation and reference with max time "
            "diff {max_diff} (s) and time offset {offset_2} (s)"}

    if num_matches < threshold * max_pairs:
        warnings.warn("Alert !!!!!!!!!!!!!!!!!!!!!!! \
                       The estimated trajectory has not enough \
                       timestamps within the GT timestamps. \
                       May be not be enough for aligned and not accurate results.",
                      category=Warning, stacklevel=2)

    print(f"Found {num_matches} of maximum {max_pairs} possible matching "
          f"timestamps between estimation and reference with maximum time "
          f"diff.: {max_diff} (s) and time offset: {offset_2} (s).")

    return traj_ref_aligned, traj_est_aligned

def process_data(traj_ref, traj_est,
                 pose_type: PoseRelaType = 'translation',
                 metric_type: MericType = 'ape'):
    r'''
    To get the error of the pose based on the pose type
    Args:
        traj_ref: StampedSE3
            The trajectory for reference.
        traj_est: StampedSE3
            The trajectory for estimation.
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
    if metric_type == 'ape':
        if pose_type == 'translation':
            E = traj_est.translation() - traj_ref.translation()
        else:
            E = (traj_est.poses.Inv() @ traj_ref.poses).matrix()
    elif metric_type == 'rpe':
        E = (traj_ref.poses.Inv() @ traj_est.poses).matrix()
    print(f"Compared {len(E)} pose pairs.")

    if pose_type == 'translation':
        if metric_type == 'ape':
            return torch.linalg.norm(E, dim=-1)
        elif metric_type == 'rpe':
            return torch.tensor([torch.linalg.norm(E_i[:3, 3]) for E_i in E])
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

def get_pairs_by_frames(traj, delta, all_pairs=False):
    r'''
    Get index of pairs in the trajectory by its index distance
    Args:
        traj: StampedSE3
            The trajectory
        delta: float
            The delta to select the pair.
        all_pairs: bool
            If True, all pairs will be used for evaluation.
    Returns: list
        id_pairs: list of index pairs
    '''
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

def get_pairs_by_translation(traj, delta, tol=0.0, all_pairs=False):
    r'''
    Get index of pairs in the trajectory by its path distance in meters
    Args:
        traj: StampedSE3
            The trajectory
        delta: float
            The delta to select the pair.
        tol: float
            Absolute path tolerance to accept or reject pairs in all_pairs mode.
        all_pairs: bool
            If True, all pairs will be used for evaluation.
    Returns: list
        id_pairs: list of index pairs
    '''
    if all_pairs:
        idx_0 = []
        idx_1 = []
        distances = traj.accumulated_distances
        for i in range(distances.size(0) - 1):
            offset = i + 1
            distance_from_here = distances[offset:] - distances[i]
            candidate_index = torch.argmin(torch.abs(distance_from_here - delta)).item()
            if (torch.abs(distance_from_here[candidate_index] - delta) > tol):
                continue
            idx_0.append(i)
            idx_1.append(candidate_index + offset)
        id_pairs = (idx_0, idx_1)
    else:
        idx = []
        previous_translation = traj.translation()[0]
        current_path = 0.0
        for i, current_translation in enumerate(traj.translation()):
            current_path += float(torch.norm(current_translation - previous_translation))
            previous_translation = current_translation
            if current_path >= delta:
                idx.append(i)
                current_path = 0.0
        id_pairs = (idx[:-1], idx[1:])
    return id_pairs

def id_pairs_from_delta(traj, delta=1.0,
                        delta_unit: Unit='frame', rel_tol=0.1,
                        all_pairs= False):
    r'''
    Get index of pairs with distance==delta from a trajectory
    Args:
        traj: StampedSE3
            The trajectory
        delta: float
            The delta to select the pair.
        delta_unit: Unit
            The type of the relation between the two trajectory.
            Including: 'frame', 'meters'
        rel_tol: float
            Relative tolerance to accept or reject deltas.
        all_pairs: bool
            If True, all pairs will be used for evaluation.
    Returns: list
        id_pairs: list of index pairs
    '''
    if delta_unit == 'frame':
        id_pairs = get_pairs_by_frames(traj, int(delta), all_pairs)
    elif delta_unit == 'meters':
        id_pairs = get_pairs_by_translation(traj, delta, delta * rel_tol, all_pairs)
    else:
        raise ValueError(f"unsupported delta unit: {delta_unit}")

    if len(id_pairs) == 0:
        ValueError(
            f"delta = {delta} ({delta_unit.value}) produced an empty index list - "
            "try lower values or a less strict tolerance")

    print(
        f"Found {len(id_pairs[0])} pairs with delta {delta} ({delta_unit}) "
        f"among {traj.num_poses} poses " +
        ("using consecutive pairs." if not all_pairs else "using all pairs."))

    return id_pairs

def get_result(error) -> dict:
    '''
    statistical data of the error
    '''
    result_dict ={}
    result_dict['max']    = torch.max(error.abs()).item()
    result_dict['mean']   = torch.mean(error.abs()).item()
    result_dict['median'] = torch.median(error.abs()).item()
    result_dict['min']    = torch.min(error.abs()).item()
    result_dict['rmse']   = torch.sqrt(torch.mean(torch.pow(error, 2))).item()
    result_dict['sse']    = torch.sum(torch.pow(error, 2)).item()
    result_dict['std']    = torch.std(error.abs()).item()

    return result_dict

def compute_APE(traj_ref, traj_est, pose_relation: PoseRelaType='translation',
                max_diff=0.01, offset_2=0.0, align=False, with_scale=False,
                n_to_align=-1, align_origin=False, match_thresh = 0.3):
    r'''
    Computes the Absolute Pose Error (RPE) between two trajectories
    Args:
        traj_ref: StampedSE3
            The reference trajectory
        traj_est: StampedSE3
            The estimated trajectory
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
        with_scale: bool
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
    print("Synchronizing trajectories...")
    traj_ref, traj_est = associate_trajectories(traj_ref, traj_est, max_diff, offset_2, match_thresh)
    trans_mat = identity_Sim3(1, dtype=traj_est.dtype, device=traj_est.device)

    # Align the trajectories.
    if align or with_scale:
        print("Aligning using Umeyama's method..." +
              (" (with scale correction)" if with_scale else ""))
        n_to_align = traj_est.num_poses if n_to_align == -1 else n_to_align
        est_trans = traj_est.translation()[:,:n_to_align]
        ref_trans = traj_ref.translation()[:,:n_to_align]
        trans_mat = svdstf(est_trans, ref_trans, with_scale)
    elif align_origin:
        trans_mat[..., :7] = (traj_ref.first_pose @ traj_est.first_pose.Inv()).data

    traj_est.align(trans_mat)

    error = process_data(traj_ref, traj_est, pose_relation, metric_type = 'ape')
    result = get_result(error)

    return result

def compute_RPE(traj_ref, traj_est, pose_relation: PoseRelaType='translation',
                max_diff=0.01, offset_2=0.0, align=False, with_scale=False,
                n_to_align=-1, align_origin=False, delta_unit: Unit='frame',
                delta=1.0, rel_delta_tol=0.1, all_pairs=False,
                match_thresh=0.3, pairs_from_reference=False):
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
        with_scale: bool
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
        rel_delta_tol: float
            Relative tolerance to accept or reject deltas.
        all_pairs: bool
            If True, all pairs will be used for evaluation.
        match_thresh: float
            The threshold for the matching pair.
            i.e. If the matching pairs are under the threshold,
                 the warning will given.
        pairs_from_reference: bool
            If True, reference trajectory will be used for get index of pairs.
    Returns: dict
        error: The statics error of the trajectory
    '''
    print("Synchronizing trajectories...")
    traj_ref, traj_est = associate_trajectories(traj_ref, traj_est, max_diff, offset_2, match_thresh)
    trans_mat = identity_Sim3(1, dtype=traj_est.dtype, device=traj_est.device)

    # Align the trajectories.
    if align or with_scale:
        print("Aligning using Umeyama's method..." +
              (" (with scale correction)" if with_scale else ""))
        n_to_align = traj_est.num_poses if n_to_align == -1 else n_to_align
        est_trans = traj_est.translation()[:,:n_to_align]
        ref_trans = traj_ref.translation()[:,:n_to_align]
        trans_mat = svdstf(est_trans, ref_trans, with_scale)
    elif align_origin:
        trans_mat[...,:7] = (traj_ref.first_pose @ traj_est.first_pose.Inv()).data

    traj_est.align(trans_mat)

    sour_id, tar_id = id_pairs_from_delta(
            (traj_ref if pairs_from_reference else traj_est), delta,
            delta_unit, rel_delta_tol, all_pairs)

    pose_ref_rela = traj_ref[sour_id].poses.Inv() @ traj_ref[tar_id].poses
    pose_est_rela = traj_est[sour_id].poses.Inv() @ traj_est[tar_id].poses
    traj_ref_rela = StampedSE3(pose_ref_rela, traj_ref[sour_id].timestamps)
    traj_est_rela = StampedSE3(pose_est_rela, traj_est[sour_id].timestamps)

    error = process_data(traj_ref_rela, traj_est_rela, pose_relation, metric_type = 'rpe')
    result = get_result(error)

    return result
