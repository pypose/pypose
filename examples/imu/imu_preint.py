import os
import torch
import pykitti
import argparse
import numpy as np
import pypose as pp
from datetime import datetime
import torch.utils.data as Data
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.collections import PatchCollection


class KITTI_IMU(Data.Dataset):
    def __init__(self, root, dataname, drive):
        super().__init__()
        self.data = pykitti.raw(root, dataname, drive)

    def __len__(self):
        return len(self.data.timestamps) - 1

    def __getitem__(self, i):
        dt = torch.tensor([datetime.timestamp(self.data.timestamps[i+1]) - datetime.timestamp(self.data.timestamps[i])])
        ang = torch.tensor([self.data.oxts[i].packet.wx, self.data.oxts[i].packet.wy, self.data.oxts[i].packet.wz])
        acc = torch.tensor([self.data.oxts[i].packet.ax, self.data.oxts[i].packet.ay, self.data.oxts[i].packet.az])
        vel = torch.tensor([self.data.oxts[i].packet.vf, self.data.oxts[i].packet.vl, self.data.oxts[i].packet.vu])
        rot = pp.euler2SO3([self.data.oxts[i].packet.roll, self.data.oxts[i].packet.pitch, self.data.oxts[i].packet.yaw])
        pos_gt = self.data.oxts[i].T_w_imu[0:3, 3]
        return dt, ang, acc, vel, rot, pos_gt

    def init_value(self):
        P = torch.tensor(self.data.oxts[0].T_w_imu[:3,3])
        R = pp.mat2SO3(torch.tensor(self.data.oxts[0].T_w_imu[:3,:3]))
        V = R @ torch.tensor([self.data.oxts[0].packet.vf, self.data.oxts[0].packet.vl, self.data.oxts[0].packet.vu])
        return P.unsqueeze(0), R.unsqueeze(0), V.unsqueeze(0)


def plot_gaussian(ax, means, covs, color=None, sigma=3):
    ''' Set specific color to show edges, otherwise same with facecolor.'''
    ellipses = []
    for i in range(len(means)):
        eigvals, eigvecs = np.linalg.eig(covs[i])
        axis = np.sqrt(eigvals) * sigma
        slope = eigvecs[1][0] / eigvecs[1][1]
        angle = 180.0 * np.arctan(slope) / np.pi
        ellipses.append(Ellipse(means[i, 0:2], axis[0], axis[1], angle=angle))
    ax.add_collection(PatchCollection(ellipses, edgecolors=color, linewidth=1))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='IMU Preintegration')
    parser.add_argument("--device", type=str, default='cpu', help="cuda or cpu")
    parser.add_argument("--integrating-step", type=int, default=1, help="number of integrated steps")
    parser.add_argument("--batch-size", type=int, default=1, help="batch size, only support 1 now")
    parser.add_argument("--save", type=str, default='./examples/imu/save/', help="location of png files to save")
    parser.add_argument("--dataroot", type=str, default='./examples/imu/', help="dataset location downloaded")
    parser.add_argument("--dataname", type=str, default='2011_09_26', help="dataset name")
    parser.add_argument("--datadrive", nargs='+', type=str, default=["0001","0002","0005","0009","0011",
                        "0013","0014","0015","0017","0018","0019","0020","0022","0005"], help="data sequences")
    parser.add_argument('--plot3d', dest='plot3d', action='store_true', help="plot in 3D space, default: False")
    parser.set_defaults(plot3d=False)
    args = parser.parse_args(); print(args)
    os.makedirs(os.path.join(args.save), exist_ok=True)
    torch.set_default_tensor_type(torch.DoubleTensor)

    for drive in args.datadrive:
        dataset = KITTI_IMU(args.dataroot, args.dataname, drive)
        p, r, v = dataset.init_value()
        integrator = pp.module.IMUPreintegrator(p, r, v).to(args.device)
        loader = Data.DataLoader(dataset=dataset, batch_size=args.batch_size)
        poses, poses_gt = [p.to(args.device)], [p.to(args.device)]
        covs = [torch.zeros(9, 9, device=args.device)]
        for idx, (dt, ang, acc, vel, rot, pos_gt) in enumerate(loader):
            dt,  ang = dt.to(args.device),  ang.to(args.device)
            acc, rot = acc.to(args.device), rot.to(args.device)
            poses_gt.append(pos_gt.to(args.device))
            integrator.update(dt, ang, acc, rot)
            if idx % args.integrating_step == 0:
                states = integrator()
                poses.append(states['pos'])
                covs.append(states['cov'])
        poses = torch.cat(poses).cpu().numpy()
        poses_gt = torch.cat(poses_gt).cpu().numpy()
        covs = torch.stack(covs).cpu().numpy()

        plt.figure(figsize=(5, 5))
        if args.plot3d:
            ax = plt.axes(projection='3d')
            ax.plot3D(poses[:,0], poses[:,1], poses[:,2], 'b')
            ax.plot3D(poses_gt[:,0], poses_gt[:,1], poses_gt[:,2], 'r')
        else:
            ax = plt.axes()
            ax.plot(poses[:,0], poses[:,1], 'b')
            ax.plot(poses_gt[:,0], poses_gt[:,1], 'r')
            plot_gaussian(ax, poses[:, 0:2], covs[:, 6:8,6:8])
        plt.title("PyPose IMU Integrator")
        plt.legend(["PyPose", "Ground Truth"])
        figure = os.path.join(args.save, args.dataname+'_'+drive+'.png')
        plt.savefig(figure)
        print("Saved to", figure)
