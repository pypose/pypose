import torch
from torch import nn

from .. import se3
from .. import mat2SE3

class lieSpline(nn.Module):

    def __init__(self):
        super().__init__()
        self.B = torch.tensor([[5, 3, -3, 1],
                               [1, 3, 3, -2],
                               [0, 0, 0, 1]]) / 6

    def compute_weights(self, time):
        alpha = torch.arange(4)
        tt = time[:, None, :] ** alpha[None, :, None]
        return self.B @ tt

    def inv_SE3(self, input_Tensor):
        RT = torch.transpose(input_Tensor[:, :, :, 0:3, 0:3], 3, 4)
        p = input_Tensor[:, :, :, 0:3, 3:]
        vec = torch.zeros(input_Tensor.shape[0],input_Tensor.shape[1],input_Tensor.shape[2], 1, 4)
        vec[:, :, :, :, 3:] = 1.0
        invT = torch.cat((torch.cat((RT, -RT @ p), dim=4), vec), dim=3)
        return invT

    def interpolateSE3(self, input_poses, time, inter_begin_and_end = True):
        """ B spline interpolation in SE3
        Args:
            input_poses: input n(n>=4) poses with [batch_size, n, 4, 4] shape
            time: k interpolation time point with [1, 1, k] shape
        Returns:
            output_poses: output (n-3)*k poses with [batch_size, (n-3)*k, 4, 4] shape
        """
        input_poses = input_poses.matrix()
        if inter_begin_and_end:
            input_poses = torch.cat(
                [torch.cat([input_poses[:, [0], :, :], input_poses], dim=1), input_poses[:, [-1], :, :]], dim=1)

        timeSize = time.shape[-1]
        batchSize = input_poses.shape[0]
        posesSize = input_poses.shape[1]
        w = self.compute_weights(time)
        w0 = w[0, :, 0, :].T
        w1 = w[0, :, 1, :].T
        w2 = w[0, :, 2, :].T

        posesTensor = torch.stack([input_poses[:, i:i + 4, :, :] for i in range(posesSize - 3)], dim=1)

        T_delta = input_poses[:, 0:-3, :, :]
        T_delta = T_delta.unsqueeze(2)
        T_delta = T_delta.repeat(1, 1, timeSize, 1, 1)
        A0 = se3(w0 * mat2SE3(self.inv_SE3(posesTensor[:, :, [0], :, :])
                                          @ posesTensor[:, :, [1], :, :]).Log()).matrix()
        A1 = se3(w1 * mat2SE3(self.inv_SE3(posesTensor[:, :, [1], :, :])
                                          @ posesTensor[:, :, [2], :, :]).Log()).matrix()
        A2 = se3(w2 * mat2SE3(self.inv_SE3(posesTensor[:, :, [2], :, :])
                                          @ posesTensor[:, :, [3], :, :]).Log()).matrix()

        waypoints = (T_delta @ A0 @ A1 @ A2).reshape((batchSize, timeSize * (posesSize - 3), 4, 4))

        waypoints = mat2SE3(waypoints)
        return waypoints






