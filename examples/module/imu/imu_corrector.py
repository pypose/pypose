import torch
import pypose as pp
from torch import nn
import tqdm, argparse
import torch.utils.data as Data
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from imu_dataset import KITTI_IMU, imu_collate, move_to


class IMUIntegrator(nn.Module):
    def __init__(self):
        super().__init__()
        self.imu = pp.module.IMUPreintegrator(reset=True, prop_cov=False)

    def forward(self, data, init_state):
        if self.eval:
            rot = None
        else:
            rot = data['gt_rot'].contiguous()

        return self.imu(init_state = init_state, dt = data['dt'], gyro = data['gyro'],
            acc = data['acc'], rot = rot)

class IMUCorrector(nn.Module):
    def __init__(self, size_list= [6, 128, 128, 128, 6]):
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
        corrected_acc = output[...,:3] * 0.1 + data["acc"]
        corrected_gyro = output[...,3:] * 0.1 + data["gyro"]
        if self.eval:
            rot = None
        else:
            rot = data['gt_rot'].contiguous()

        return self.imu(init_state = init_state, dt = data['dt'], gyro = corrected_gyro,
            acc = corrected_acc, rot = rot)


def plot_trajectory(poses, poses_gt, poses_corrected, save = False):
    plt.figure(figsize=(5, 5))
    ax = plt.axes()
    ax.plot(poses[:,0], poses[:,1], 'b')
    ax.plot(poses_corrected[:,0], poses_corrected[:,1], 'g')
    ax.plot(poses_gt[:,0], poses_gt[:,1], 'r')
    plt.title("PyPose IMU Integrator")
    plt.legend(["Integration", "Corrected", "Ground Truth"])

    if save:
        plt.savefig("imu_corrector.png")
    plt.show()


def get_loss(state, data):
    pos_loss = torch.nn.functional.mse_loss(state['pos'], data['gt_pos'], reduction='mean')
    rot_loss = (data['gt_rot'] * state['rot'].Inv()).Log().norm(dim=-1).mean()

    loss = pos_loss + rot_loss * 5e2
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
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    return (running_loss/(i+1))


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

    return (running_loss/(i+1))

def evaluate(model, init, loader, device = "cuda:0"):
    poses, poses_gt = [], []
    model.eval()
    with torch.no_grad():
        for idx, data in enumerate(loader):
            data = move_to(data, args.device)
            state = model(data, init)
            init = {
                "pos": state['pos'][..., -1, :],
                "rot": state['rot'][..., -1, :],
                "vel": state['vel'][..., -1, :]}

            poses_gt.append(data['gt_pos'][..., -1, :].cpu())
            poses.append(state['pos'][..., -1, :].cpu())

    poses_gt = torch.cat(poses_gt).numpy()
    poses = torch.cat(poses).numpy()

    return poses, poses_gt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    parser.add_argument("--device", type=str, default=device, help="cuda or cpu")
    parser.add_argument("--batch-size", type=int, default=16, help="batch size")
    parser.add_argument("--max_epoches", type=int, default=50, help="max_epoches")
    parser.add_argument("--duration", type=int, default=100, help="sequence length")
    parser.add_argument("--dataroot", type=str, default='./examples/module/imu', \
                        help="dataset location downloaded")
    parser.add_argument("--dataname", type=str, default='2011_09_26', help="dataset name")
    parser.add_argument("--datadrive", nargs='+', type=str, default=["0022"],
                        help="data sequences")
    args = parser.parse_args(); print(args)

    train_dataset = KITTI_IMU(args.dataroot, args.dataname, args.datadrive[0],
                              duration=args.duration, mode='train', download=True)
    test_dataset = KITTI_IMU(args.dataroot, args.dataname, args.datadrive[0],
                             duration=args.duration, mode='test', download=True)
    evaluate_dataset = KITTI_IMU(args.dataroot, args.dataname, args.datadrive[0],
                            duration=10, step_size=10, mode='evaluate', download=True)

    train_loader = Data.DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                                   collate_fn=imu_collate, shuffle=True)
    test_loader = Data.DataLoader(dataset=test_dataset, batch_size=args.batch_size,
                                  collate_fn=imu_collate, shuffle=False)
    evaluate_loader = Data.DataLoader(dataset=evaluate_dataset, batch_size=1,
                                      collate_fn=imu_collate, shuffle=False)

    network = IMUCorrector().to(args.device)
    integrator = IMUIntegrator().to(args.device)

    optimizer = torch.optim.Adam(network.parameters(), lr = 5e-5)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor = 0.1, patience = 15)

    best_loss = torch.inf
    para_stae_dict = None
    for epoch_i in range(args.max_epoches):
        train_loss = train(network, train_loader, epoch_i, optimizer, device=args.device)
        test_loss = test(network, test_loader, device = args.device)
        reference_loss = test(integrator, train_loader, device = args.device)
        scheduler.step(test_loss)
        if best_loss > test_loss:
            best_loss = test_loss
            para_stae_dict = network.state_dict()
        print("train loss: %f test loss: %f reference loss: %f"
                %(train_loss, test_loss, reference_loss))

    print("evaluating...")

    init = evaluate_dataset.get_init_value()
    network = IMUCorrector().to(args.device)
    network.load_state_dict(para_stae_dict)
    poses_corrected, poses_gt = evaluate(network, init, evaluate_loader, device=args.device)
    poses, _ = evaluate(integrator, init, evaluate_loader, device = args.device)

    plot_trajectory(poses, poses_gt, poses_corrected, save = True)
