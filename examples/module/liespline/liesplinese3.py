import os.path
import argparse
import pypose as pp
import matplotlib.pyplot as plt
from pytransform3d.rotations import plot_basis
from pypose.module.liesplinese3 import *

def plot_result(wayposes, x_range, y_range, z_range, k = 0, oriposes = None, save=None, show=None):
    assert k < wayposes.shape[0]
    wayposes = wayposes.matrix()
    ax = None
    for pose in wayposes[k, :, :, :]:
        R = pose[0:3, 0:3]
        p = pose[0:3, 3]
        ax = plot_basis(ax=ax, s=0.15, R=R, p=p)
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_zlim(z_range)
    if oriposes != None:
        oriposes = oriposes.matrix()
        for pose in oriposes[k, :, :, :]:
            R = pose[0:3, 0:3]
            p = pose[0:3, 3]
            ax = plot_basis(ax=ax, s=0.3, R=R, p=p, color="b")
    if save is not None:
        file_path = os.path.join(save, 'liespline.png')
        plt.savefig(file_path)
        print("Save to", file_path)
    if show:
        plt.show()

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='LieSpline Example')
    parser.add_argument("--device", type=str, default='cpu', help="cuda or cpu")
    parser.add_argument("--save", type=str, default='./examples/module/liespline/save/',
                        help="location of png files to save")
    parser.add_argument('--show', dest='show', action='store_true',
                        help="show plot, default: False")
    parser.set_defaults(show=True)
    args = parser.parse_args()
    os.makedirs(os.path.join(args.save), exist_ok=True)
    print(args)
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


    wayposes = BSlpineSE3(input_poses, time)
    plot_result(wayposes, [0, 5.], [0, 5.], [0, 1.2], oriposes= input_poses, save=args.save, show=args.show)
