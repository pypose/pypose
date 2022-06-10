import os, glob
import torch
import torch.nn as nn
import numpy as np
import pypose as pp

import torch.utils.data as Data
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import argparse

import tqdm
import pykitti
from datetime import datetime


class MLPcorrection(torch.nn.Module):
    def __init__(self, size_list= [6, 64, 128, 128, 128, 6]):
        super(MLPcorrection, self).__init__()
        layers = []
        self.size_list = size_list
        for i in range(len(size_list) - 2):
            layers.append(nn.Linear(size_list[i],size_list[i+1]))
            # layers.append(nn.BatchNorm1d(size_list[i+1]))
            layers.append(nn.GELU())
        layers.append(nn.Linear(size_list[-2], size_list[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, data):
        feature = torch.cat([data["feat_acc"], data["feat_gyro"]], dim = -1)
        B, F = feature.shape[:2]

        init_state = {
            "p": data['gt_pos'][:,:1,:], 
            "r": data['gt_rot'][:,:1,:],
            "v": data['gt_vel'][:,:1,:],}
        output = self.net(feature.reshape(B*F,6)).reshape(B, F, 6)
        corrected_acc = output[...,:3] + data["feat_acc"]
        corrected_gyro= output[...,3:] + data["feat_gyro"]

        inte_state, _ = pp.module.IMUPreintegrator.batch_imu_integrate(init_state, data['dt'], corrected_acc, corrected_gyro, data['gt_rot'][:,:-1].contiguous())
        return inte_state, _


def move_to(obj, device):
    if torch.is_tensor(obj):return obj.to(device)
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


def get_loss(inte_state, cov_state, data):
    pos_loss = torch.nn.functional.mse_loss(inte_state['pos'], data['gt_pos'][:,1:,:])
    rot_loss = torch.nn.functional.mse_loss(inte_state['rot'].Log(), data['gt_rot'][:,1:,:].Log())

    loss = pos_loss + rot_loss
    return loss, {'pos_loss': pos_loss, 'rot_loss': rot_loss}


class KITTI_IMU(Data.Dataset):
    def __init__(self, root, dataname, drive, duration = 10, step_size = 1, mode = 'train',):
        super().__init__()
        self.duration = duration
        self.data = pykitti.raw(root, dataname, drive)
        self.seq_len = len(self.data.timestamps) - 1

        self.dt = torch.tensor([datetime.timestamp(self.data.timestamps[i+1]) - datetime.timestamp(self.data.timestamps[i]) for i in range(self.seq_len)])
        self.gyro = torch.tensor([[self.data.oxts[i].packet.wx, self.data.oxts[i].packet.wy, self.data.oxts[i].packet.wz] for i in range(self.seq_len)])
        self.acc = torch.tensor([[self.data.oxts[i].packet.ax, self.data.oxts[i].packet.ay, self.data.oxts[i].packet.az] for i in range(self.seq_len)])
        self.gt_rot = pp.euler2SO3(torch.tensor([[self.data.oxts[i].packet.roll, self.data.oxts[i].packet.pitch, self.data.oxts[i].packet.yaw] for i in range(self.seq_len)]))
        self.gt_vel = self.gt_rot @ torch.tensor([[self.data.oxts[i].packet.vf, self.data.oxts[i].packet.vl, self.data.oxts[i].packet.vu] for i in range(self.seq_len)])
        self.gt_pos = torch.tensor([self.data.oxts[i].T_w_imu[0:3, 3] for i in range(self.seq_len)] )

        start_frame = 0
        end_frame = self.seq_len
        if mode == 'train':
            end_frame = np.floor(self.seq_len * 0.5).astype(int)
        elif mode == 'test':
            start_frame = np.floor(self.seq_len * 0.5).astype(int)

        self.index_map = [i for i in range(0, end_frame - start_frame - self.duration, step_size)]

    def __len__(self):
        return len(self.index_map)
        
    def __getitem__(self, i):
        frame_id = self.index_map[i]
        end_frame_id = frame_id + self.duration
        return {
            'dt': self.dt[frame_id: end_frame_id],
            'feat_acc': self.acc[frame_id: end_frame_id],
            'feat_gyro': self.gyro[frame_id: end_frame_id],
            'gt_pos': self.gt_pos[frame_id: end_frame_id+1],
            'gt_rot': self.gt_rot[frame_id: end_frame_id+1],
            'gt_vel': self.gt_vel[frame_id: end_frame_id+1],
        }


def custom_collate(data):
    feat_acc = torch.stack([torch.tensor(d['feat_acc']) for d in data])
    feat_gyro = torch.stack([torch.tensor(d['feat_gyro']) for d in data])

    gt_pos = torch.stack([d['gt_pos'] for d in data])
    gt_rot = torch.stack([d['gt_rot'] for d in data])
    gt_vel = torch.stack([d['gt_vel'] for d in data])

    dt = torch.stack([d['dt'] for d in data]).unsqueeze(-1)
    
    return {
        'dt': dt,
        'feat_acc': feat_acc,
        'feat_gyro': feat_gyro,
        'gt_pos': gt_pos,
        'gt_vel': gt_vel,
        'gt_rot': gt_rot,
    }


def do_train(network, train_loader, epoch, optimizer, device="cuda:0"):
    """
    Train network for one epoch using a specified data loader
    Outputs all targets, predicts, predicted covariance params, and losses in numpy arrays
    """
    network.train()
    running_loss = 0
    t_range = tqdm.tqdm(train_loader)
    for i, data in enumerate(t_range):
        data = move_to(data, device)
        inte_state, cov_state = network(data)

        losses, _ = get_loss(inte_state, cov_state, data)
        running_loss += losses.item()

        t_range.set_description(f'iteration: {i:04d}, losses: {losses:.06f}')
        t_range.refresh()
        losses.backward()
        optimizer.step()

    return (running_loss/i)


def test(network, loader, device = "cuda:0"):
    network.eval()
    with torch.no_grad():
        running_loss = 0
        for i, data in enumerate(tqdm.tqdm(loader)):
            data = move_to(data, device)
            inte_state, cov_state = network(data)

            losses, _ = get_loss(inte_state, cov_state, data)
            running_loss += losses.item()

        print("the running loss of the test set %0.6f"%(running_loss/i))

    return (running_loss/i)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/base.conf', help='config file path')
    parser.add_argument("--device", type=str, default='cuda:0', help="cuda or cpu")
    parser.add_argument("--batch-size", type=int, default=1, help="batch size, only support 1 now")
    parser.add_argument("--window_size", type=int, default=10, help="window_size")
    parser.add_argument("--max_epoches", type=int, default=100, help="max_epoches")
    parser.add_argument("--dataroot", type=str, default='/home/yuheng/data/KITTI_raw', help="dataset location downloaded")
    parser.add_argument("--dataname", type=str, default='2011_09_26', help="dataset name")
    parser.add_argument("--datadrive", nargs='+', type=str, default=[ "0001"], help="data sequences")
    parser.add_argument('--load_ckpt', default=False, action="store_true")

    args = parser.parse_args(); print(args)
    base_confs = {"device": args.device, }
    
    train_dataset = KITTI_IMU(args.dataroot, args.dataname, args.datadrive[0])
    test_dataset = KITTI_IMU(args.dataroot, args.dataname, args.datadrive[0], mode="test")
    train_loader = Data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, collate_fn=custom_collate, shuffle=True)
    test_loader = Data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, collate_fn=custom_collate, shuffle=False)

    ## optimizer
    network = MLPcorrection().to(args.device)
    optimizer = torch.optim.Adam(network.parameters(), lr = 5e-6)  # to use with ViTs
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor = 0.1, patience = 10)# default setup

    best_loss = np.inf
    for epoch_i in range(args.max_epoches):
        train_loss = do_train(network, train_loader, epoch_i, optimizer, device = args.device)
        test_loss = test(network, test_loader, device = args.device)
        scheduler.step(train_loss)
        print("train loss:%f test loss%f "%(train_loss, test_loss))