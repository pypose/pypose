import torch, warnings
from ..function.geometry import svdstf
from ..lietensor import mat2SO3, SE3, Sim3, identity_Sim3
from ..lietensor.lietensor import SE3Type, Sim3Type


class StampedSE3(object):
    def __init__(self, timestamps=None, poses_SE3=None, dtype=torch.float64):
        r"""
        Internal class for represent the trajectory with timestamps.

        Args:
            timestamps (array-like of ``float`` or ``None``):
                The timestamps of the trajectory.
                Must have same length with poses.
                For example, `torch.tensor(...)` or `None`.
            poses_SE3 (array-like of ``SE3``):
                The trajectory poses. Must be SE3.
                For example, `pypose.SE3(torch.rand(10, 7))`.
            dtype (`torch.float64` or higher):
                The data type for poses to calculate.
                Defaults to torch.float64.

        Returns:
            None.

        Error:
            1. ValueError: The poses have shape problem.
            2. ValueError: The timestamps have shape problem.
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
        stamps_1 (array-like of ``float``):
            First list of timestamps.
            For example, `torch.tensor(...)`.
        stamps_2 (array-like of ``float``):
            Second list of timestamps.
            For example, `torch.tensor(...)`.
        max_diff (``float``, optional):
            The maximum allowed absolute time difference (in seconds)
            for associating poses. Defaults to 0.01.
        offset_2 (``float``, optional):
            The aligned offset (in seconds) for the second timestamps.
            Defaults to 0.0.

    Returns:
        matching_indices_1: ``list[int]``
            Indices of timestamps_1 that match with timestamps_1.
        matching_indices_2: ``list[int]``
            Indices of timestamps_2 that match with timestamps_2.
    """
    stamps_2 += offset_2
    diff_mat = (stamps_1[..., None] - stamps_2[None]).abs()
    indices_1 = torch.arange(len(stamps_1), device=stamps_1.device)
    value, indices_2 = diff_mat.min(dim=-1)

    matching_indices_1 = indices_1[value < max_diff].tolist()
    matching_indices_2 = indices_2[value < max_diff].tolist()

    return matching_indices_1, matching_indices_2


def associate_traj(rtraj, etraj, max_diff=0.01, offset_2=0.0, threshold=0.3):
    r"""
    Associate two trajectories by matching their timestamps.

    Args:
        rtraj (``StampedSE3``):
            The trajectory for reference.
        etraj (``StampedSE3``):
            The trajectory for estimation.
        max_diff (``float``, optional):
            The maximum allowed absolute time difference (in seconds)
            for associating poses. Defaults to 0.01.
        offset_2 (``float``, optional):
            The aligned offset (in seconds) for the second timestamps.
            Defaults to 0.0.
        threshold (``float``, optional):
            The threshold for valid matching pairs. If the ratio of
            matching pairs is below this threshold, a warning is issued.
            Defaults to 0.3.

    Returns:
        etraj_aligned: ``StampedSE3``
            The aligned estimation trajectory.
        rtraj_aligned: ``StampedSE3``
            The aligned reference trajectory.

    Warning:
        The matched stamps are under the threshold.
    """
    snd_longer = len(etraj.timestamps) > len(rtraj.timestamps)
    traj_long = etraj if snd_longer else rtraj
    traj_short = rtraj if snd_longer else etraj
    max_pairs = len(traj_short.timestamps)

    matching_indices_short, matching_indices_long = matching_time_indices(
        traj_short.timestamps, traj_long.timestamps, max_diff,
        offset_2 if snd_longer else -offset_2)

    assert len(matching_indices_short) == len(matching_indices_long), \
        {r"matching_time_indices returned unequal number of indices"}

    num_matches = len(matching_indices_long)
    traj_short = traj_short[matching_indices_short]
    traj_long = traj_long[matching_indices_long]

    rtraj_aligned = traj_short if snd_longer else traj_long
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

    return rtraj_aligned, etraj_aligned


def compute_error(rtraj, etraj, output: str = 'translation', mtype: str = 'ape', otype:str = 'All'):
    r'''
    Get the error of the pose based on the output type.

    Args:
        rtraj (``StampedSE3``):
            The reference trajectory.
        etraj (``StampedSE3``):
            The estimated trajectory.
        output: (``str``, optional):
            The type of pose error.
            Including: 'translation', 'rotation', 'pose', 'rotation', 'radian', 'degree'.
        mtype: (``str``, optional):
            The type of metrics.
            Including: 'ape', 'rpe'.

    Returns:
        error: The all errors of the pose.

    Error:
        ValueError: The output type is not supported.

    Remarks:
        if mtype == 'ape'
            'translation': || t_{e} - t_{r} ||_2
            'rotation': || R_{e}^T * R_{r} - I_3 ||_2
            'pose':  || T_{e}^{-1} * T_{r} - I_4 ||_2
            'radian': :math:`|| \mathrm{Log}(R_{e}^T * R_{r}) ||_2`
            'degree': :math:`\mathrm{Degree}(|| \mathrm{Log}(R_{e}^T * R_{r}) ||_2)`
        if mtype == 'rpe':
            'translation': :math:`|| -{R_{r}^{rel}}^T * t_{r}^{rel} + {R_{e}^{rel}}^T * t_{e}^{rel} ||_2`
            'rotation': :math:`|| {R_{r}^{rel}}^T * R_{e}^{rel} - I_3 ||_2`
            'pose': :math:`|| {T_{r}^{rel}}^{-1} * T_{e}^{rel} - I_4 ||_2`
            'radian': :math:`|| \mathrm{Log}({R_{r}^{rel}}^T * R_{e}^{rel}) ||_2)`
            'degree': :math:`\mathrm{Degree}(|| \mathrm{Log}({R_{r}^{rel}}^T * R_{e}^{rel}) ||_2))`
    '''
    if mtype == 'ape':
        if output == 'translation':
            E = etraj.translation() - rtraj.translation()
        else:
            E = (etraj.poses.Inv() @ rtraj.poses).matrix()
    elif mtype == 'rpe':
        E = (rtraj.poses.Inv() @ etraj.poses).matrix()

    if output == 'translation':
        if mtype == 'ape':
            error = torch.linalg.norm(E, dim=-1)
        elif mtype == 'rpe':
            error = E[..., :3, 3].norm(dim=-1)
    elif output == 'rotation':
        I = torch.eye(3, device=E.device, dtype=E.dtype).expand_as(E[:,:3,:3])
        error = torch.linalg.norm((E[:,:3,:3] - I), dim=(-2, -1))
    elif output == 'pose':
        I = torch.eye(4, device=E.device, dtype=E.dtype).expand_as(E)
        error = torch.linalg.norm((E - I), dim=(-2, -1))
    elif output == 'radian':
        error = mat2SO3(E[:,:3,:3], check=False).Log().norm(dim=-1)
    elif output == 'degree':
        error = mat2SO3(E[:,:3,:3], check=False).Log().norm(dim=-1).rad2deg()
    else:
        raise ValueError(f"Unknown output type: {output}")

    options = ['All', 'Max', 'Min', 'Mean', 'Median', 'RMSE', 'SSE', 'STD']
    if otype not in options:
        raise ValueError(f"Unknown output metric type, select one in {options}")
    results = {}
    if otype == 'Max' or 'All':
        results['Max'] = torch.max(error.abs())
    if otype == 'Min' or 'All':
        results['Min'] = torch.min(error.abs())
    if otype == 'Mean' or 'All':
        results['Mean'] = torch.mean(error.abs())
    if otype == 'Median' or 'All':
        results['Median'] = torch.median(error.abs())
    if otype == 'RMSE' or 'All':
        results['RMSE'] = torch.sqrt(torch.mean(torch.pow(error, 2)))
    if otype == 'SSE' or 'All':
        results['SSE'] = torch.sum(torch.pow(error, 2))
    if otype == 'STD' or 'All':
        results['STD']  = torch.std(error.abs())

    if otype == 'All':
        return results # return a dict
    else:
        return results[otype] # return a tensor


def pairs_by_frames(traj, delta, all=False):
    r'''
    Get index of pairs in the trajectory by its index distance.

    Args:
        traj (``StampedSE3``):
            The trajectory.
        delta (``float``):
            The interval step used to select the pair.
        all (``bool``, optional):
            If True, all associated pairs are used for evaluation.
            Defaults to False.

    Returns: list
        id_pairs: list of index pairs.
    '''
    traj_len = traj.num_poses
    delta = int(delta)
    assert delta >= 1, "delta must >= 1"
    if all:
        ids_1 = torch.arange(traj_len, device=traj.device, dtype=torch.long)
        ids_2 = ids_1 + delta
        id_pairs = (ids_1[ids_2<traj_len].tolist(),
                    ids_2[ids_2<traj_len].tolist())
    else:
        ids = torch.arange(0, traj_len, delta, device=traj.device, dtype=torch.long)
        id_pairs = (ids[:-1].tolist(), ids[1:].tolist())

    return id_pairs


def pairs_by_dist(traj, delta, tol=0.0, all=False):
    r'''
    Get index of pairs in the trajectory by its path distance.

    Args:
        traj (``StampedSE3``):
            The trajectory
        delta (``float``):
            The interval step used to select the pair.
        tol (``float``, optional):
            The absolute path tolerance for accepting or rejecting pairs in all mode.
            Defaults to 0.0.
        all (``bool``, optional):
            If True, all associated pairs are used for evaluation.
            Defaults to False.

    Returns: list
        id_pairs: list of index pairs.
    '''
    if all:
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


def pair_id(traj, delta=1.0, associate: str='frame', rtol=0.1, all= False):
    r'''
    Get index of pairs with distance==delta from a trajectory.

    Args:
        traj (``StampedSE3``):
            The trajectory.
        delta (``float``, optional):
            The interval step used to select the pair. For example, when
            `associate='distance'`, it can represent the distance
            step in meters. Defaults to 1.0.
        associate (``str``, optional):
            The method used to associate pairs between the two trajectories.
            Supported options: 'frame', 'distance'. Defaults to 'frame'.
        rtol (``float``, optional):
            The relative tolerance for accepting or rejecting deltas.
            Defaults to 0.1.
        all (``bool``, optional):
            If True, all associated pairs are used for evaluation.
            Defaults to False.

    Returns:
        list: list of index pairs.
    '''
    if associate == 'frame':
        id_pairs = pairs_by_frames(traj, int(delta), all)
    elif associate == 'distance':
        id_pairs = pairs_by_dist(traj, delta, delta * rtol, all)
    else:
        raise ValueError(f"unsupported delta unit: {associate}")

    if len(id_pairs) == 0:
        raise ValueError(
            f"delta = {delta} ({associate}) produced an empty index list - "
            "try lower values or a less strict tolerance")

    return id_pairs


def ape(rstamp, rpose, estamp, epose, etype: str = "translation", diff: float = 0.01,
        offset: float = 0.0, align: bool = False, scale: bool = False, nposes: int = -1,
        origin: bool = False, thresh: float = 0.3, otype: str = 'All'):
    r'''
    Compute the Absolute Pose Error (APE) between two trajectories.

    Args:
        rstamp (array-like of ``float`` or ``None``):
            The timestamps of the reference trajectory.
            Must have the same length as `rpose`.
            For example, `torch.tensor([...])` or `None`.
        rpose (array-like of ``SE3``):
            The poses of the reference trajectory.
            Must have the same length as `rstamp`.
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

            ``'translation'``: :math:`|| t_{e} - t_{r} ||_2`

            ``'rotation'``: :math:`|| R_{e}^T * R_{r} - I_3 ||_2`

            ``'pose'``: :math:`|| T_{e}^{-1} * T_{r} - I_4 ||_2`

            ``'radian'``: :math:`|| \mathrm{Log}(R_{e}^T * R_{r}) ||_2`

            ``'degree'``: :math:`\mathrm{Degree}(|| \mathrm{Log}(R_{e}^T * R_{r}) ||_2)`

            Default: ``'translation'``

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
        otype (``str``, optional):
            The output type for the metric. Supported options include:

            ``'All'``: All metrics will be computed and returned.

            ``'Max'``: The Max error is computed and returned.

            ``'Min'``: The Min error is computed and returned.

            ``'Mean'``: The Mean error is computed and returned.

            ``'Median'``: The Median error is computed and returned.

            ``'RMSE'``: The root mean square error (RMSE) is computed and returned.

            ``'SSE'``: The sum of square error (SSE) is computed and returned.

            ``'STD'``: The standard deviation (STD) is computed and returned.

            Defaults to ``'All'``.

    Returns:
        ``dict`` or ``Tensor``: The computed statistics of the APE (Absolute Pose Error).
        The return is a ``dict`` if ``otype`` is not ``'all'``, otherwise a ``Tensor``.

    Examples:
        >>> import torch
        >>> import pypose as pp
        >>> rstamp = torch.tensor([1311868163.8696999550, 1311868163.8731000423,
        ...                        1311868163.8763999939])
        >>> rpose = pp.SE3([[-0.1357000023, -1.4217000008,  1.4764000177,
        ...                   0.6452999711, -0.5497999787,  0.3362999856, -0.4101000130],
        ...                 [-0.1357000023, -1.4218000174,  1.4764000177,
        ...                   0.6453999877, -0.5497000217,  0.3361000121, -0.4101999998],
        ...                 [-0.1358000040, -1.4219000340,  1.4764000177,
        ...                   0.6455000043, -0.5498999953,  0.3357999921, -0.4101000130]])
        >>> estamp = torch.tensor([1311868164.3631811142, 1311868164.3990259171,
        ...                        1311868164.4309399128])
        >>> epose = pp.SE3([[0.0000000000, 0.0000000000, 0.0000000000,
        ...                  0.0000000000, 0.0000000000, 0.0000000000, 1.0000000000],
        ...                 [-0.0005019300, 0.0010138600, -0.0020097860,
        ...                  -0.0020761820,-0.0010706080, -0.0007627490, 0.9999969602],
        ...                 [0.0004298200, 0.0019603260, -0.0048985220,
        ...                 -0.0043526068,-0.0036625920, -0.0023494449, 0.9999810457]])
        >>> pp.metric.ape(rstamp, rpose, estamp, epose)
        {'Max': tensor(2.0590, dtype=torch.float64),
         'Min': tensor(2.0541, dtype=torch.float64),
         'Mean': tensor(2.0565, dtype=torch.float64),
         'Median': tensor(2.0562, dtype=torch.float64),
         'RMSE': tensor(2.0565, dtype=torch.float64),
         'SSE': tensor(12.6871, dtype=torch.float64),
         'STD': tensor(0.0025, dtype=torch.float64)}
        >>> pp.metric.ape(rstamp, rpose, estamp, epose, otype='Mean')
        tensor(2.0565, dtype=torch.float64)
    '''
    rtraj, etraj = StampedSE3(rstamp, rpose), StampedSE3(estamp, epose)
    rtraj, etraj = associate_traj(rtraj, etraj, diff, offset, thresh)
    trans_mat = identity_Sim3(1, dtype=etraj.dtype, device=etraj.device)

    if align or scale:
        nposes = etraj.num_poses if nposes == -1 else nposes
        est_trans = etraj.translation()[..., :nposes]
        ref_trans = rtraj.translation()[..., :nposes]
        trans_mat = svdstf(est_trans, ref_trans, scale)
    elif origin:
        trans_mat[..., :7] = (rtraj.first_pose @ etraj.first_pose.Inv()).data

    etraj.align(trans_mat)

    return compute_error(rtraj, etraj, etype, mtype = 'ape', otype=otype)


def rpe(rstamp, rpose, estamp, epose, etype: str = "translation", diff: float = 0.01,
        offset: float = 0.0, align: bool = False, scale: bool = False, nposes: int = -1,
        origin: bool = False, associate: str = 'frame', delta: float = 1.0, rtol: float = 0.1,
        all: bool = False, thresh: float = 0.3, rpair: bool = False, otype: str = 'All'):
    r'''
    Compute the Relative Pose Error (RPE) between two trajectories.

    Args:
        rstamp (array-like of ``float`` or ``None``):
            The timestamps of the reference trajectory.
            Must have the same length as `rpose`.
            For example, `torch.tensor([...])` or `None`.
        rpose (array-like of ``SE3``):
            The poses of the reference trajectory.
            Must have the same length as `rstamp`.
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

            ``'translation'``: :math:`||{R_{r}^{rel}}^T * t_{r}^{rel} - {R_{e}^{rel}}^T * t_{e}^{rel}||_2`

            ``'rotation'``: :math:`|| {R_{r}^{rel}}^T * R_{e}^{rel} - I_3 ||_2`

            ``'pose'``: :math:`|| {T_{r}^{rel}}^{-1} * T_{e}^{rel} - I_4 ||_2`

            ``'radian'``: :math:`|| \mathrm{Log}({R_{r}^{rel}}^T * R_{e}^{rel}) ||_2`

            ``'degree'``: :math:`\mathrm{Degree}(|| \mathrm{Log}({R_{r}^{rel}}^T * R_{e}^{rel}) ||_2))`

            Default: ``'translation'``

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
            The interval step used to select the pair. For example, when
            `associate='distance'`, it can represent the distance
            step in meters. Defaults to 1.0.
        rtol (``float``, optional):
            The relative tolerance for accepting or rejecting deltas.
            Defaults to 0.1.
        all (``bool``, optional):
            If True, all associated pairs are used for evaluation.
            Defaults to False.
        thresh (``float``, optional):
            The threshold for valid matching pairs. If the ratio of
            matching pairs is below this threshold, a warning is issued.
            Defaults to 0.3.
        rpair (``bool``, optional):
            Use reference trajectory to compute the pairing indices or not. Defaults to False.
        otype (``str``, optional):
            The output type for the metric. Supported options include:

            ``'All'``: All metrics will be computed and returned.

            ``'Max'``: The Max error is computed and returned.

            ``'Min'``: The Min error is computed and returned.

            ``'Mean'``: The Mean error is computed and returned.

            ``'Median'``: The Median error is computed and returned.

            ``'RMSE'``: The root mean square error (RMSE) is computed and returned.

            ``'SSE'``: The sum of square error (SSE) is computed and returned.

            ``'STD'``: The standard deviation (STD) is computed and returned.

            Defaults to ``'All'``.

    Returns:
        ``dict`` or ``Tensor``: The computed statistics of the RPE (Relative Pose Error).
        The return is a ``dict`` if ``otype`` is not ``'all'``, otherwise a ``Tensor``.

    Examples:
        >>> import torch
        >>> import pypose as pp
        >>> rstamp = torch.tensor([1311868163.8696999550, 1311868163.8731000423,
        ...                        1311868163.8763999939])
        >>> rpose = pp.SE3([[-0.1357000023, -1.4217000008,  1.4764000177,
        ...                   0.6452999711, -0.5497999787,  0.3362999856, -0.4101000130],
        ...                 [-0.1357000023, -1.4218000174,  1.4764000177,
        ...                    0.6453999877, -0.5497000217,  0.3361000121, -0.4101999998],
        ...                 [-0.1358000040, -1.4219000340,  1.4764000177,
        ...                   0.6455000043, -0.5498999953,  0.3357999921, -0.4101000130]])
        >>> estamp = torch.tensor([1311868164.3631811142, 1311868164.3990259171,
        ...                        1311868164.4309399128])
        >>> epose = pp.SE3([[0.0000000000, 0.0000000000, 0.0000000000,
        ...                  0.0000000000, 0.0000000000, 0.0000000000, 1.0000000000],
        ...                 [-0.0005019300, 0.0010138600, -0.0020097860,
        ...                  -0.0020761820,-0.0010706080, -0.0007627490, 0.9999969602],
        ...                 [0.0004298200, 0.0019603260, -0.0048985220,
        ...                 -0.0043526068, -0.0036625920, -0.0023494449, 0.9999810457]])
        >>> pp.metric.rpe(rstamp, rpose, estamp, epose)
        {'Max': tensor(0.0032, dtype=torch.float64),
         'Min': tensor(0.0023, dtype=torch.float64),
         'Mean': tensor(0.0027, dtype=torch.float64),
         'Median': tensor(0.0023, dtype=torch.float64),
         'RMSE': tensor(0.0028, dtype=torch.float64),
         'SSE': tensor(1.5428e-05, dtype=torch.float64),
         'STD': tensor(0.0006, dtype=torch.float64)}
        >>> pp.metric.rpe(rstamp, rpose, estamp, epose, otype='Mean')
        tensor(0.0027, dtype=torch.float64)
    '''
    rtraj, etraj = StampedSE3(rstamp, rpose), StampedSE3(estamp, epose)
    rtraj, etraj = associate_traj(rtraj, etraj, diff, offset, thresh)
    trans_mat = identity_Sim3(1, dtype=etraj.dtype, device=etraj.device)

    if align or scale:
        nposes = etraj.num_poses if nposes == -1 else nposes
        est_trans = etraj.translation()[:,:nposes]
        ref_trans = rtraj.translation()[:,:nposes]
        trans_mat = svdstf(est_trans, ref_trans, scale)
    elif origin:
        trans_mat[...,:7] = (rtraj.first_pose @ etraj.first_pose.Inv()).data

    etraj.align(trans_mat)

    sour_id, tar_id = pair_id((rtraj if rpair else etraj), delta, associate, rtol, all)

    rpose_rela = rtraj[sour_id].poses.Inv() @ rtraj[tar_id].poses
    epose_rela = etraj[sour_id].poses.Inv() @ etraj[tar_id].poses
    rtraj_rela = StampedSE3(rtraj[sour_id].timestamps, rpose_rela)
    etraj_rela = StampedSE3(etraj[sour_id].timestamps, epose_rela)

    return compute_error(rtraj_rela, etraj_rela, etype, mtype = 'rpe', otype = otype)
