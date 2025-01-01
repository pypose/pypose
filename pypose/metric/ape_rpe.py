import warnings
from typing import TypedDict, Literal

import torch
from ..function.geometry import svdstf
from ..lietensor import mat2SO3, SE3, Sim3, identity_Sim3
from ..lietensor.lietensor import SE3Type, Sim3Type


class StampedSE3(object):
    def __init__(self, timestamps=None, poses_SE3=None, dtype=torch.float64):
        r"""
        Internal class for represent the trajectory with timestamps.
        Args:
            timestamps: The timestamps of the trajectory.
                        Must have same length with poses.
                    e.g torch.tensor(...) or None
            poses_SE3: The trajectory poses. Must be SE3
                    e.g. pypose.SE3(torch.rand(10, 7))
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
        return StampedSE3(self.timestamps[index], self.poses[index], self.poses.dtype)

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
        zeros = torch.zeros(1, dtype=trans.dtype)
        norm = torch.linalg.norm(trans[:-1] - trans[1:], dim=-1, dtype=trans.dtype)
        return torch.cat((zeros, torch.cumsum(norm, dim=0)))


def matching_time_indices(stamps_1, stamps_2, max_diff=0.01, offset_2=0.0):
    r"""
    Search for the best matching timestamps of two lists of timestamps.
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


def associate_traj(ttraj, etraj, max_diff=0.01, offset_2=0.0, threshold=0.3):
    r"""
    Associate two trajectories by matching their timestamps.
    Args:
        ttraj: StampedSE3
            The trajectory for reference.
        etraj: StampedSE3
            The trajectory for estimation.
        max_diff: float
            Max allowed absolute time difference (s) for associating
        offset_2: float
            The aligned offset (s) for the second timestamps.
        threshold: float
            The threshold (%) for the matching warning
    Returns:
        etraj_aligned: StampedSE3
            The aligned estimation trajectory
        ttraj_aligned: StampedSE3
            The aligned reference trajectory
    Warning:
        The matched stamps are under the threshold
    """
    snd_longer = len(etraj.timestamps) > len(ttraj.timestamps)
    traj_long = etraj if snd_longer else ttraj
    traj_short = ttraj if snd_longer else etraj
    max_pairs = len(traj_short.timestamps)

    matching_indices_short, matching_indices_long = matching_time_indices(
        traj_short.timestamps, traj_long.timestamps, max_diff,
        offset_2 if snd_longer else -offset_2)

    assert len(matching_indices_short) == len(matching_indices_long), \
        {r"matching_time_indices returned unequal number of indices"}

    num_matches = len(matching_indices_long)
    traj_short = traj_short[matching_indices_short]
    traj_long = traj_long[matching_indices_long]

    ttraj_aligned = traj_short if snd_longer else traj_long
    etraj_aligned = traj_long if snd_longer else traj_short

    assert num_matches != 0, \
        {f"found no matching timestamps between estimation and reference with max time "
            "diff {max_diff} (s) and time offset {offset_2} (s)"}

    if num_matches < threshold * max_pairs:
        warnings.warn("Alert !!!!!!!!!!!!!!!!!!!!!!! \
                       The estimated trajectory has not enough \
                       timestamps within the GT timestamps. \
                       May be not be enough for aligned and not accurate results.",
                      category=Warning, stacklevel=2)

    return ttraj_aligned, etraj_aligned


def compute_error(ttraj, etraj, output: str = 'translation', metric_type: str = 'ape'):
    r'''
    Get the error of the pose based on the output type.
    Args:
        ttraj: StampedSE3
            The trajectory for reference.
        etraj: StampedSE3
            The trajectory for estimation.
        output: OutputType
            The type of the output error
    Returns:
        error: The all error of the pose
    Error:
        ValueError: The output type is not supported
    Remarks:
        'translation': || t_{est} - t_{ref} ||_2
        'rotation': || R_{est} - R_{ref} ||_2
        'full':  || T_{est} - T_{ref} ||_2
        'rotation_angle_rad': ||Log(R_{est} - R_{ref})||_2
        'rotation_angle_deg': Degree(||Log(R_{est} - R_{ref})||_2)
    '''
    if metric_type == 'ape':
        if output == 'translation':
            E = etraj.translation() - ttraj.translation()
        else:
            E = (etraj.poses.Inv() @ ttraj.poses).matrix()
    elif metric_type == 'rpe':
        E = (ttraj.poses.Inv() @ etraj.poses).matrix()

    if output == 'translation':
        if metric_type == 'ape':
            error = torch.linalg.norm(E, dim=-1)
        elif metric_type == 'rpe':
            error = torch.tensor([torch.linalg.norm(E_i[:3, 3]) for E_i in E])
    elif output == 'rotation':
        I = torch.eye(3, device=E.device, dtype=E.dtype).expand_as(E[:,:3,:3])
        error = torch.linalg.norm((E[:,:3,:3] - I), dim=(-2, -1))
    elif output == 'pose':
        I = torch.eye(4, device=E.device, dtype=E.dtype).expand_as(E)
        error = torch.linalg.norm((E - I), dim=(-2, -1))
    elif output == 'radian':
        error = mat2SO3(E[:,:3,:3] ).euler().norm(dim=-1)
    elif output == 'degree':
        error = (mat2SO3(E[:,:3,:3]).euler()).norm(dim=-1)
        error = torch.rad2deg(error)
    else:
        raise ValueError(f"Unknown output type: {output}")

    result_dict = {}
    result_dict['Max']    = torch.max(error.abs()).item()
    result_dict['Mean']   = torch.mean(error.abs()).item()
    result_dict['Median'] = torch.median(error.abs()).item()
    result_dict['Min']    = torch.min(error.abs()).item()
    result_dict['RMSE']   = torch.sqrt(torch.mean(torch.pow(error, 2))).item()
    result_dict['SSE']    = torch.sum(torch.pow(error, 2)).item()
    result_dict['STD']    = torch.std(error.abs()).item()

    return result_dict


def pairs_by_frames(traj, delta, use_all=False):
    r'''
    Get index of pairs in the trajectory by its index distance.
    Args:
        traj: StampedSE3
            The trajectory
        delta: float
            The delta to select the pair.
        use_all: bool
            If True, all pairs will be used for evaluation.
    Returns: list
        id_pairs: list of index pairs
    '''
    traj_len = traj.num_poses
    delta = int(delta)
    assert delta >= 1, "delta must >= 1"
    if use_all:
        ids_1 = torch.arange(traj_len, device=traj.device, dtype=torch.long)
        ids_2 = ids_1 + delta
        id_pairs = (ids_1[ids_2<traj_len].tolist(),
                    ids_2[ids_2<traj_len].tolist())
    else:
        ids = torch.arange(0, traj_len, delta, device=traj.device, dtype=torch.long)
        id_pairs = (ids[:-1].tolist(), ids[1:].tolist())

    return id_pairs


def pairs_by_dist(traj, delta, tol=0.0, use_all=False):
    r'''
    Get index of pairs in the trajectory by its path distance.
    Args:
        traj: StampedSE3
            The trajectory
        delta: float
            The delta to select the pair.
        tol: float
            Absolute path tolerance to accept or reject pairs in use_all mode.
        use_all: bool
            If True, all pairs will be used for evaluation.
    Returns: list
        id_pairs: list of index pairs
    '''
    if use_all:
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


def pair_id(traj, delta=1.0, associate: str='frame', rtol=0.1, use_all= False):
    r'''
    Get index of pairs with distance==delta from a trajectory
    Args:
        traj (``StampedSE3``):
            The trajectory
        delta (``float``):
            The delta to select the pair.
        associate (``str``):
            The type of the association relation between the two trajectory.
            Including: 'frame', 'distance'.
        rtol (``float``):
            Relative tolerance to accept or reject deltas.
        use_all (``bool``):
            If True, all pairs will be used for evaluation.
    Returns:
        list: list of index pairs
    '''
    if associate == 'frame':
        id_pairs = pairs_by_frames(traj, int(delta), use_all)
    elif associate == 'distance':
        id_pairs = pairs_by_dist(traj, delta, delta * rtol, use_all)
    else:
        raise ValueError(f"unsupported delta unit: {associate}")

    if len(id_pairs) == 0:
        raise ValueError(
            f"delta = {delta} ({associate}) produced an empty index list - "
            "try lower values or a less strict tolerance")

    return id_pairs


def ape(tstamp, tpose, estamp, epose, etype: str = "translation", diff: float = 0.01,
        offset: float = 0.0, align: bool = False, scale: bool = False, nposes: int = -1,
        origin: bool = False, thresh: float = 0.3):
    r'''
    Compute the Absolute Pose Error (APE) between two trajectories.

    Args:
        tstamp (array-like of ``float`` or ``None``):
            The timestamps of the true (reference) trajectory.
            Must have the same length as `tpose`.
            For example, `torch.tensor([...])` or `None`.
        tpose (array-like of ``SE3``):
            The poses of the true (reference) trajectory.
            Must have the same length as `tstamp`.
            For example, `pypose.SE3(torch.rand(10, 7))`.
        estamp (array-like of ``float`` or ``None``):
            The timestamps of the estimated trajectory.
            Must have the same length as `epose`.
            For example, `torch.tensor([...])` or `None`.
        epose (array-like of ``SE3``):
            The poses of the estimated trajectory.
            Must have the same length as `estamp`.
            For example, `pypose.SE3(torch.rand(10, 7))`.
        etype (``str``, optional):
            The type of pose error. Supported options include:

            'translation': :math:`|| t_{est} - t_{ref} ||_2`

            'rotation': :math:`|| R_{est} - R_{ref} ||_2`

            'pose': :math:`|| T_{est} - T_{ref} ||_2`

            'radian': :math:`||\mathrm{Log}(R_{est} - R_{ref})||_2`

            'degree': :math:`\mathrm{Degree}(||Log(R_{est} - R_{ref})||_2)`

        diff (``float``, optional):
            The maximum allowed absolute time difference (in seconds)
            for associating poses. Defaults to 0.01.
        offset (``float``, optional):
            The aligned offset (in seconds) for the second timestamps.
            Defaults to 0.0.
        align (``bool``, optional):
            If True, the trajectory is aligned using a scaled SVD method.
            Defaults to False.
        scale (``bool``, optional):
            If True, the scale is corrected using the SVD method.
            Defaults to False.
        nposes (``int``, optional):
            The number of poses to use for alignment. If -1, all
            poses are used. Defaults to -1.
        origin (``bool``, optional):
            If True, the trajectory is aligned by the first pose.
            Defaults to False.
        thresh (``float``, optional):
            The threshold for valid matching pairs. If the ratio of
            matching pairs is below this threshold, a warning is issued.
            Defaults to 0.3.

    Return:
        dict: The computed statistics of the APE (Absolute Pose Error).
    '''
    ttraj, etraj = StampedSE3(tstamp, tpose), StampedSE3(estamp, epose)
    ttraj, etraj = associate_traj(ttraj, etraj, diff, offset, thresh)
    trans_mat = identity_Sim3(1, dtype=etraj.dtype, device=etraj.device)

    if align or scale:
        nposes = etraj.num_poses if nposes == -1 else nposes
        est_trans = etraj.translation()[..., :nposes]
        ref_trans = ttraj.translation()[..., :nposes]
        trans_mat = svdstf(est_trans, ref_trans, scale)
    elif origin:
        trans_mat[..., :7] = (ttraj.first_pose @ etraj.first_pose.Inv()).data

    etraj.align(trans_mat)

    return compute_error(ttraj, etraj, etype, metric_type = 'ape')


def rpe(tstamp, tpose, estamp, epose, etype: str = "translation", diff: float = 0.01,
        offset: float = 0.0, align: bool = False, scale: bool = False, nposes: int = -1,
        origin: bool = False, associate: str = 'frame', delta: float = 1.0, rtol: float = 0.1,
        use_all: bool = False, thresh: float = 0.3, tpair: bool = False):
    r'''
    Compute the Relative Pose Error (RPE) between two trajectories.

    Args:
        tstamp (array-like of ``float`` or ``None``):
            The timestamps of the true (reference) trajectory.
            Must have the same length as `tpose`.
            For example, `torch.tensor([...])` or `None`.
        tpose (array-like of ``SE3``):
            The poses of the true (reference) trajectory.
            For example, `pypose.SE3(torch.rand(10, 7))`.
        estamp (array-like of ``float`` or ``None``):
            The timestamps of the estimated trajectory.
            Must have the same length as `epose`.
            For example, `torch.tensor([...])` or `None`.
        epose (array-like of ``SE3``):
            The poses of the estimated trajectory.
            For example, `pypose.SE3(torch.rand(10, 7))`.
        etype (``str``, optional):
            The type of pose error. Supported options include:

            'translation': :math:`|| t_{est} - t_{ref} ||_2`

            'rotation': :math:`|| R_{est} - R_{ref} ||_2`

            'pose': :math:`|| T_{est} - T_{ref} ||_2`

            'radian': :math:`||\mathrm{Log}(R_{est} - R_{ref})||_2`

            'degree': :math:`\mathrm{Degree}(||Log(R_{est} - R_{ref})||_2)`
        diff (``float``, optional):
            The maximum allowed absolute time difference (in seconds)
            for associating poses. Defaults to 0.01.
        offset (``float``, optional):
            The aligned offset (in seconds) for the second timestamps.
            Defaults to 0.0.
        align (``bool``, optional):
            If True, the trajectory is aligned using a scaled SVD method.
            Defaults to False.
        scale (``bool``, optional):
            If True, the scale is corrected using the SVD method.
            Defaults to False.
        nposes (``int``, optional):
            The number of poses to use for alignment. If -1, all
            poses are used. Defaults to -1.
        origin (``bool``, optional):
            If True, the trajectory is aligned by the first pose.
            Defaults to False.
        associate (``str``, optional):
            The method used to associate pairs between the two trajectories.
            Supported options: 'frame', 'distance'. Defaults to 'frame'.
        delta (``float``, optional):
            The delta used to select the pair. For example, when
            `associate='distance'`, it can represent the distance
            step in meters. Defaults to 1.0.
        rtol (``float``, optional):
            The relative tolerance for accepting or rejecting deltas.
            Defaults to 0.1.
        use_all (``bool``, optional):
            If True, all associated pairs are used for evaluation.
            Defaults to False.
        thresh (``float``, optional):
            The threshold for valid matching pairs. If the ratio of
            matching pairs is below this threshold, a warning is issued.
            Defaults to 0.3.
        tpair (``bool``, optional):
            Use true trajectory to compute the pairing indices or not. Defaults to False.
    
    Return:
        dict: The computed statistics of the RPE (Relative Pose Error).
    '''
    ttraj, etraj = StampedSE3(tstamp, tpose), StampedSE3(estamp, epose)
    ttraj, etraj = associate_traj(ttraj, etraj, diff, offset, thresh)
    trans_mat = identity_Sim3(1, dtype=etraj.dtype, device=etraj.device)

    if align or scale:
        nposes = etraj.num_poses if nposes == -1 else nposes
        est_trans = etraj.translation()[:,:nposes]
        ref_trans = ttraj.translation()[:,:nposes]
        trans_mat = svdstf(est_trans, ref_trans, scale)
    elif origin:
        trans_mat[...,:7] = (ttraj.first_pose @ etraj.first_pose.Inv()).data

    etraj.align(trans_mat)

    sour_id, tar_id = pair_id((ttraj if tpair else etraj), delta, associate, rtol, use_all)

    tpose_rela = ttraj[sour_id].poses.Inv() @ ttraj[tar_id].poses
    epose_rela = etraj[sour_id].poses.Inv() @ etraj[tar_id].poses
    ttraj_rela = StampedSE3(ttraj[sour_id].timestamps, tpose_rela)
    etraj_rela = StampedSE3(etraj[sour_id].timestamps, epose_rela)

    return compute_error(ttraj_rela, etraj_rela, etype, metric_type = 'rpe')
