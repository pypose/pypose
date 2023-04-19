import torch
from torch import nn

class lieSpline(nn.Module):

    def __init__(self):
        super().__init__()
        
    def compute_weights(self, time):
        alpha = torch.arange(4, dtype=time.dtype, device=time.device)
        tt = time[:, None, :] ** alpha[None, :, None]
        B = torch.tensor([[5, 3, -3, 1],
                               [1, 3, 3, -2],
                               [0, 0, 0, 1]], dtype=time.dtype, device=time.device) / 6
        return B @ tt

    def interpolateSE3(self, input_poses, time, inter_begin_and_end = True):
        """ B spline interpolation in SE3
        Args:
            input_poses: input n(n>=4) poses with [batch_size, n, 4, 4] shape
            time: k interpolation time point with [1, 1, k] shape
            inter_begin_and_end: the flag of whether interpolate the poses between input_poses[0] and input_poses[1] and input_poses[-2] and input_poses[-1]
        Returns:
            output_poses: output (n-3)*k poses with [batch_size, (n-3)*k, 4, 4] shape
        """
        if inter_begin_and_end:
            input_poses = torch.cat(
                [torch.cat([input_poses[:, [0], :], input_poses], dim=1), input_poses[:, [-1], :]], dim=1)

        timeSize = time.shape[-1]
        batchSize = input_poses.shape[0]
        posesSize = input_poses.shape[1]
        w = self.compute_weights(time)
        w0 = w[0, :, 0, :].T
        w1 = w[0, :, 1, :].T
        w2 = w[0, :, 2, :].T

        posesTensor = torch.stack([input_poses[:, i:i + 4, :] for i in range(posesSize - 3)], dim=1)

        T_delta = input_poses[:, 0:-3, :]
        T_delta = T_delta.unsqueeze(2)
        T_delta = T_delta.repeat(1, 1, timeSize, 1)
        A0 = ((posesTensor[:, :, [0], :].Inv()*(posesTensor[:, :, [1], :])).Log() * w0).Exp()
        A1 = ((posesTensor[:, :, [1], :].Inv()*(posesTensor[:, :, [2], :])).Log() * w1).Exp()
        A2 = ((posesTensor[:, :, [2], :].Inv()*(posesTensor[:, :, [3], :])).Log() * w2).Exp()
        waypoints = T_delta * A0 * A1* A2
        return waypoints





