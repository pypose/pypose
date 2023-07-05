import torch
import os.path
import argparse
import pypose as pp
import matplotlib.pyplot as plt
from pytransform3d.rotations import plot_basis


def plot_result(wayposes, xrange, yrange, zrange, k = 0,
                oriposes = None, save=None, show=None):
    assert k < wayposes.shape[0]
    wayposes = wayposes.matrix()
    fig = plt.figure(figsize=(20, 10))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax1.view_init(30, 30)
    ax2.view_init(30, 30)
    ax1.set_xlim(xrange)
    ax1.set_ylim(yrange)
    ax1.set_zlim(zrange)
    ax2.set_xlim(xrange)
    ax2.set_ylim(yrange)
    ax2.set_zlim(zrange)
    x = torch.zeros(wayposes.shape[1], dtype=wayposes.dtype, device=wayposes.device)
    y = torch.zeros(wayposes.shape[1], dtype=wayposes.dtype, device=wayposes.device)
    z = torch.zeros(wayposes.shape[1], dtype=wayposes.dtype, device=wayposes.device)
    i = 0
    for pose in wayposes[k, :, :, :]:
        R = pose[0:3, 0:3]
        p = pose[0:3, 3]
        ax1 = plot_basis(ax=ax1, s=0.15, R=R, p=p)
        x[i], y[i], z[i] = p[0], p[1], p[2]
        i += 1
    ax1.plot3D(x, y, z, c='orangered')
    ax2.plot3D(x, y, z, linewidth=2.0)
    if oriposes != None:
        ax2.scatter(oriposes[k, :, 0], oriposes[k, :, 1], oriposes[k, :, 2],
                    c="r", marker="*", linewidth=3.0)
    if save is not None:
        file_path = os.path.join(save, 'Bspline.png')
        plt.savefig(file_path)
        print("Save to", file_path)
    if show:
        plt.show()

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='LieSpline Example')
    parser.add_argument("--device", type=str, default='cpu', help="cuda or cpu")
    parser.add_argument("--save", type=str, default='./examples/module/spline/save/',
                        help="location of png files to save")
    parser.add_argument('--show', dest='show', action='store_true',
                        help="show plot, default: False")
    parser.set_defaults(show=True)
    args = parser.parse_args()
    os.makedirs(os.path.join(args.save), exist_ok=True)
    print(args)
    angle1 = pp.euler2SO3(torch.Tensor([0., 0., 0.]))
    angle2 = pp.euler2SO3(torch.Tensor([torch.pi / 4., torch.pi / 3., torch.pi / 2.]))
    poses = pp.LieTensor([[[0., 4., 0., angle1[0], angle1[1], angle1[2], angle1[3]],
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
                   [3., 0., 5., angle2[0], angle2[1], angle2[2], angle2[3]]]],
                 ltype=pp.SE3_type)
    wayposes = pp.bspline(poses, 0.1, True)
    xrange = [0, 5.]
    yrange = [0, 5.]
    zrange = [0, 1.2]
    plot_result(wayposes, xrange, yrange, zrange,
                oriposes=poses, save=args.save, show=args.show)
