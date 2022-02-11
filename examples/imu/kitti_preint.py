import torch
import pykitti
import datetime
import pypose as pp
import matplotlib.pyplot as plt
import torch.utils.data as Data
torch.set_default_tensor_type(torch.DoubleTensor)


class KITTI_IMU(Data.Dataset):
	def __init__(self, root, dataname, drive):
		super().__init__()
		self.data = pykitti.raw(root, dataname, drive)

	def __len__(self):
		return len(self.data.timestamps)-1

	def __getitem__(self, i):
		dt = datetime.datetime.timestamp(self.data.timestamps[i+1]) - datetime.datetime.timestamp(self.data.timestamps[i])
		ang = torch.tensor([self.data.oxts[i].packet.wx, self.data.oxts[i].packet.wy, self.data.oxts[i].packet.wz])
		acc = torch.tensor([self.data.oxts[i].packet.ax, self.data.oxts[i].packet.ay, self.data.oxts[i].packet.az])
		vel = torch.tensor([self.data.oxts[i].packet.vf, self.data.oxts[i].packet.vl, self.data.oxts[i].packet.vu])
		rot = pp.euler2SO3([self.data.oxts[i].packet.roll, self.data.oxts[i].packet.pitch, self.data.oxts[i].packet.yaw])
		pos_gt = self.data.oxts[i].T_w_imu[0:3,3]
		return dt, ang, acc, vel, rot, pos_gt

	def init_value(self):
		P = torch.tensor(self.data.oxts[0].T_w_imu[:3,3])
		R = pp.mat2SO3(torch.tensor(self.data.oxts[0].T_w_imu[:3,:3]))
		V = R @ torch.tensor([self.data.oxts[0].packet.vf, self.data.oxts[0].packet.vl, self.data.oxts[0].packet.vu])
		return P, R, V


if __name__ == '__main__':

	# Change this to directory where to save plots
	plot_dir = './examples/imu/plot/'
	# Change this to the directory where you store KITTI data
	basedir = './examples/imu'
	# Specify the dataset to load
	datename = '2011_09_26'
	drives = ["0001","0002","0005","0009","0011","0013","0014","0015","0017","0018","0019","0020","0022","0005"]
	device = 'cuda'

	for drive in drives:

		dataset = KITTI_IMU(basedir, datename, drive)
		p, r, v = dataset.init_value()
		integrator = pp.IMUPreintegrator(p, r, v).to(device)
		loader = Data.DataLoader(dataset=dataset)
		poses, poses_gt = [p.view(1,-1).to(device)], [p.view(1,-1).to(device)]
		for idx, (dt, ang, acc, vel, rot, pos_gt) in enumerate(loader):
			dt, ang, acc, rot = dt.to(device), ang.to(device), acc.to(device), rot.to(device)
			integrator.update(dt, ang, acc, rot)
			pos, rot, vel = integrator()
			poses.append(pos)
			poses_gt.append(pos_gt.to(device))

		poses = torch.cat(poses).cpu().numpy()
		poses_gt = torch.cat(poses_gt).cpu().numpy()
		fig = plt.figure(figsize=(15, 15))
		ax = plt.axes(projection='3d')
		ax.plot3D(poses[:,0], poses[:,1], poses[:,2], 'b')
		ax.plot3D(poses_gt[:,0], poses_gt[:,1], poses_gt[:,2], 'r')
		plt.legend(["pypose", "ground_truth"])
		figure = plot_dir + drive +'.png'
		plt.savefig(figure)
		# plt.show()
