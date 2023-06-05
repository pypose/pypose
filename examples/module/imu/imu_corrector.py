import torch
import pypose as pp
from torch import nn
import tqdm, argparse
import torch.utils.data as Data
from torch.optim.lr_scheduler import ReduceLROnPlateau
from imu_dataset import KITTI_IMU, imu_collate, move_to


class IMUCorrector(nn.Module):
    def __init__(self, size_list= [6, 64, 128, 128, 128, 6]):
        super().__init__()
        layers = []
        self.size_list = size_list
        for i in range(len(size_list) - 2):
            layers.append(nn.Linear(size_list[i],size_list[i+1]))
            layers.append(nn.GELU())
        layers.append(nn.Linear(size_list[-2], size_list[-1]))
        self.net = nn.Sequential(*layers)
        self.imu = pp.module.IMUPreintegrator(reset=True, prop_cov=False)

    def forward(self, data, init_state):
        feature = torch.cat([data["acc"], data["gyro"]], dim = -1)
        B, F = feature.shape[:2]

        output = self.net(feature.reshape(B*F,6)).reshape(B, F, 6)
        corrected_acc = output[...,:3] + data["acc"]
        corrected_gyro = output[...,3:] + data["gyro"]

        return self.imu(init_state = init_state, dt = data['dt'], gyro = corrected_gyro,
            acc = corrected_acc, rot = data['gt_rot'].contiguous())


def get_loss(state, data):
    pos_loss = torch.nn.functional.mse_loss(state['pos'][:,-1,:], data['gt_pos'][:,-1,:])
    rot_loss = (data['gt_rot'][:,-1,:] * state['rot'][:,-1,:].Inv()).Log().norm()

    loss = pos_loss + rot_loss
    return loss, {'pos_loss': pos_loss, 'rot_loss': rot_loss}


def train(network, train_loader, epoch, optimizer, device="cuda:0"):
    """
    Train network for one epoch using a specified data loader
    Outputs all targets, predicts, predicted covariance params, and losses in numpy arrays
    """
    network.train()
    running_loss = 0
    t_range = tqdm.tqdm(train_loader)
    for i, data in enumerate(t_range):
        data = move_to(data, device)
        init_state = {
            "pos": data['init_pos'],
            "rot": data['init_rot'][:,:1,:],
            "vel": data['init_vel'],}
        state = network(data, init_state)

        losses, _ = get_loss(state, data)
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
            init_state = {
            "pos": data['init_pos'],
            "rot": data['init_rot'][:,:1,:],
            "vel": data['init_vel'],}
            state = network(data, init_state)

            losses, _ = get_loss(state, data)
            running_loss += losses.item()

        print("the running loss of the test set %0.6f"%(running_loss/i))

    return (running_loss/i)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    parser.add_argument("--device", type=str, default=device, help="cuda or cpu")
    parser.add_argument("--batch-size", type=int, default=4, help="batch size")
    parser.add_argument("--max_epoches", type=int, default=100, help="max_epoches")
    parser.add_argument("--dataroot", type=str, default='./examples/module/imu', \
                        help="dataset location downloaded")
    parser.add_argument("--dataname", type=str, default='2011_09_26', help="dataset name")
    parser.add_argument("--datadrive", nargs='+', type=str, default=[ "0001"], \
                        help="data sequences")
    parser.add_argument('--load_ckpt', default=False, action="store_true")
    args = parser.parse_args(); print(args)

    train_dataset = KITTI_IMU(args.dataroot, args.dataname, args.datadrive[0],
                              duration=10, mode='train', download=True)
    test_dataset = KITTI_IMU(args.dataroot, args.dataname, args.datadrive[0],
                             duration=10, mode='test', download=True)
    train_loader = Data.DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                                   collate_fn=imu_collate, shuffle=True)
    test_loader = Data.DataLoader(dataset=test_dataset, batch_size=args.batch_size,
                                  collate_fn=imu_collate, shuffle=False)

    ## optimizer
    network = IMUCorrector().to(args.device)
    optimizer = torch.optim.Adam(network.parameters(), lr=5e-6)  # to use with ViTs
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10)

    for epoch in range(args.max_epoches):
        train_loss = train(network, train_loader, epoch, optimizer, device = args.device)
        test_loss = test(network, test_loader, device = args.device)
        scheduler.step(train_loss)
        print("train loss: %f test loss: %f "%(train_loss, test_loss))
