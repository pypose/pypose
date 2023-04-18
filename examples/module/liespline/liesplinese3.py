import torch
import pypose as pp
import matplotlib.pyplot as plt
from pytransform3d.rotations import plot_basis
from pypose.module.liesplinese3 import lieSpline

angle1 = pp.euler2SO3(torch.Tensor([0., 0., 0.]))
angle2 = pp.euler2SO3(torch.Tensor([torch.pi / 4., torch.pi / 3., torch.pi / 2.]))
time = torch.arange(0, 1, 0.25).reshape(1, 1, -1)
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
                             [3., 0., 5., angle2[0], angle2[1], angle2[2], angle2[3]]]], ltype=pp.SE3_type)

LS = lieSpline()
waypoints = LS.interpolateSE3(input_poses, time)
print(waypoints.shape)
wayposes = waypoints.matrix()
print(wayposes.shape)
ax = None
for pose in wayposes[0, :, :, :]:

    R = pose[0:3, 0:3]
    p = pose[0:3, 3]
    ax = plot_basis(ax=ax, s=0.15, R=R, p=p)
ax.set_xlim([0,5.])
ax.set_ylim([0,5.])
ax.set_zlim([0,1.2])

plt.savefig("liespline.png")
plt.show()
