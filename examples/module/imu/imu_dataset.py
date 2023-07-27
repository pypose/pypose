import torch
import pykitti
import numpy as np
import pypose as pp
from datetime import datetime
import torch.utils.data as Data
from torchvision.datasets.utils import download_and_extract_archive


class KITTI_IMU(Data.Dataset):
    datalink = 'https://github.com/pypose/pypose/releases/download/v0.2.2/2011_09_26.zip'
    def __init__(self, root, dataname, drive, duration=10, step_size=1, \
                 mode='train', download=True):
        super().__init__()
        self.duration = duration
        if download:
            download_and_extract_archive(self.datalink, root)
        self.data = pykitti.raw(root, dataname, drive)
        self.seq_len = len(self.data.timestamps) - 1
        assert mode in ['evaluate', 'train', 'test'], "{} mode unsupported.".format(mode)

        self.dt = torch.tensor([datetime.timestamp(self.data.timestamps[i+1]) -
                                datetime.timestamp(self.data.timestamps[i])
                                    for i in range(self.seq_len)])
        self.gyro = torch.tensor([[self.data.oxts[i].packet.wx,
                                   self.data.oxts[i].packet.wy,
                                   self.data.oxts[i].packet.wz]
                                   for i in range(self.seq_len)])
        self.acc = torch.tensor([[self.data.oxts[i].packet.ax,
                                  self.data.oxts[i].packet.ay,
                                  self.data.oxts[i].packet.az]
                                  for i in range(self.seq_len)])
        self.gt_rot = pp.euler2SO3(torch.tensor([[self.data.oxts[i].packet.roll,
                                                  self.data.oxts[i].packet.pitch,
                                                  self.data.oxts[i].packet.yaw]
                                                  for i in range(self.seq_len)]))
        self.gt_vel = self.gt_rot @ torch.tensor([[self.data.oxts[i].packet.vf,
                                                   self.data.oxts[i].packet.vl,
                                                   self.data.oxts[i].packet.vu]
                                                   for i in range(self.seq_len)])
        self.gt_pos = torch.tensor(np.array([self.data.oxts[i].T_w_imu[0:3, 3]
                                             for i in range(self.seq_len)]))

        start_frame = 0
        end_frame = self.seq_len
        if mode == 'train':
            end_frame = np.floor(self.seq_len * 0.5).astype(int)
        elif mode == 'test':
            start_frame = np.floor(self.seq_len * 0.5).astype(int)

        N = end_frame - start_frame - self.duration
        self.index_map = [i for i in range(0, N, step_size)]

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, i):
        frame_id = self.index_map[i]
        end_frame_id = frame_id + self.duration
        return {
            'dt': self.dt[frame_id: end_frame_id],
            'acc': self.acc[frame_id: end_frame_id],
            'gyro': self.gyro[frame_id: end_frame_id],
            'gt_pos': self.gt_pos[frame_id+1 : end_frame_id+1],
            'gt_rot': self.gt_rot[frame_id+1 : end_frame_id+1],
            'gt_vel': self.gt_vel[frame_id+1 : end_frame_id+1],
            'init_pos': self.gt_pos[frame_id][None, ...],
            'init_rot': self.gt_rot[frame_id : end_frame_id],
            # TODO: the init rotation might be used in gravity compensation
            'init_vel': self.gt_vel[frame_id][None, ...],
        }

    def get_init_value(self):
        return {'pos': self.gt_pos[:1],
                'rot': self.gt_rot[:1],
                'vel': self.gt_vel[:1]}


def imu_collate(data):
    acc = torch.stack([d['acc'] for d in data])
    gyro = torch.stack([d['gyro'] for d in data])

    gt_pos = torch.stack([d['gt_pos'] for d in data])
    gt_rot = torch.stack([d['gt_rot'] for d in data])
    gt_vel = torch.stack([d['gt_vel'] for d in data])

    init_pos = torch.stack([d['init_pos'] for d in data])
    init_rot = torch.stack([d['init_rot'] for d in data])
    init_vel = torch.stack([d['init_vel'] for d in data])

    dt = torch.stack([d['dt'] for d in data]).unsqueeze(-1)

    return {
        'dt': dt,
        'acc': acc,
        'gyro': gyro,

        'gt_pos': gt_pos,
        'gt_vel': gt_vel,
        'gt_rot': gt_rot,

        'init_pos': init_pos,
        'init_vel': init_vel,
        'init_rot': init_rot,
    }


def move_to(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    elif isinstance(obj, dict):
        res = {}
        for k, v in obj.items():
            res[k] = move_to(v, device)
        return res
    elif isinstance(obj, list):
        res = []
        for v in obj:
            res.append(move_to(v, device))
        return res
    else:
        raise TypeError("Invalid type for move_to", obj)
