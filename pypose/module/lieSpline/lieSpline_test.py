import torch
import pypose as pp
from torch import nn

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
        if inter_begin_and_end:
            print(input_poses[:,0,:,:].shape)
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
        A0 = pp.LieTensor(w0 * pp.mat2SE3(self.inv_SE3(posesTensor[:, :, [0], :, :])
                                          @ posesTensor[:, :, [1], :, :]).Log(), ltype=pp.se3_type).matrix()
        A1 = pp.LieTensor(w1 * pp.mat2SE3(self.inv_SE3(posesTensor[:, :, [1], :, :])
                                          @ posesTensor[:, :, [2], :, :]).Log(), ltype=pp.se3_type).matrix()
        A2 = pp.LieTensor(w2 * pp.mat2SE3(self.inv_SE3(posesTensor[:, :, [2], :, :])
                                          @ posesTensor[:, :, [3], :, :]).Log(), ltype=pp.se3_type).matrix()

        waypoints = (T_delta @ A0 @ A1 @ A2).reshape((batchSize, timeSize * (posesSize - 3), 4, 4))


        return waypoints





if __name__=="__main__":

    angle1 = pp.euler2SO3(torch.Tensor([0., 0., 0.]))
    angle2 = pp.euler2SO3(torch.Tensor([torch.pi / 4., torch.pi / 3., torch.pi / 2.]))
    time = torch.arange(0, 1, 0.1).reshape(1, 1, -1)
    input_poses = pp.LieTensor([[[0., 4., 0., angle1[0], angle1[1], angle1[2], angle1[3]],
                                 [0., 3., 0., angle1[0], angle1[1], angle1[2], angle1[3]],
                                 [0., 2., 0., angle1[0], angle1[1], angle1[2], angle1[3]],
                                 [0., 1., 0., angle1[0], angle1[1], angle1[2], angle1[3]],
                                 [1., 0., 1., angle2[0], angle2[1], angle2[2], angle2[3]],
                                 [2., 0., 1., angle2[0], angle2[1], angle2[2], angle2[3]],
                                 [3., 0., 1., angle2[0], angle2[1], angle2[2], angle2[3]],
                                 [4., 0., 1., angle2[0], angle2[1], angle2[2], angle2[3]]],
                                [[2., 4., 0., angle1[0], angle1[1], angle1[2], angle1[3]],
                                 [3., 3., 0., angle1[0], angle1[1], angle1[2], angle1[3]],
                                 [4., 2., 0., angle1[0], angle1[1], angle1[2], angle1[3]],
                                 [5., 1., 0., angle1[0], angle1[1], angle1[2], angle1[3]],
                                 [1., 0., 2., angle2[0], angle2[1], angle2[2], angle2[3]],
                                 [2., 0., 3., angle2[0], angle2[1], angle2[2], angle2[3]],
                                 [2., 0., 4., angle2[0], angle2[1], angle2[2], angle2[3]],
                                 [3., 0., 5., angle2[0], angle2[1], angle2[2], angle2[3]]]], ltype=pp.SE3_type).matrix()

    LS = lieSpline()
    waypoints = LS.interpolateSE3(input_poses, time)

    import numpy as np
    from util.camera_pose_visualizer import CameraPoseVisualizer

    visualizer = CameraPoseVisualizer([0, 4], [0, 4], [0, 1.2])
    for i in input_poses[0,:,:,:]:
        visualizer.extrinsic2pyramid(np.array(i), 'r', 0.1)
    for pose in waypoints[0, :, :, :]:
        visualizer.extrinsic2pyramid(np.array(pose), 'c', 0.1)
    # for pose in waypoints[1, :, :, :]:
    #     visualizer.extrinsic2pyramid(np.array(pose), 'r', 0.1)

    visualizer.show()


