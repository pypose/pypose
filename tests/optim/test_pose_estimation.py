import time
import torch
import pypose as pp
from torch import nn
import pypose.optim.kernel as ppok
from pypose.function.geometry import point2pixel, reprojerr
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Timer:
    def __init__(self):
        self.synchronize()
        self.start_time = time.time()

    def synchronize(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def tic(self):
        self.start()

    def show(self, prefix="", output=True):
        self.synchronize()
        duration = time.time()-self.start_time
        if output:
            print(prefix+"%fs" % duration)
        return duration

    def toc(self, prefix=""):
        self.end()
        return self.duration

    def start(self):
        self.synchronize()
        self.start_time = time.time()

    def end(self, reset=True):
        self.synchronize()
        self.duration = time.time()-self.start_time
        if reset:
            self.start_time = time.time()
        return self.duration


class PoseEstimation(nn.Module):
    def __init__(self, prior_pose):
        super().__init__()
        self.pose = pp.Parameter(prior_pose)

    def forward(self, intrinsics, points_3d, detected_points, prior_pose):
        prior_pose_error = (self.pose.Inv() @ prior_pose).Log().tensor()
        reprojection_error = reprojerr(points_3d, detected_points, intrinsics, self.pose)
        return prior_pose_error, reprojection_error

    def get_pose(self):
        return self.pose


if __name__ == '__main__':
    point_noise, pose_noise = 3, 0.2
    f, H, W, Np = 200, 600, 600, 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    intrinsics = torch.tensor([[f, 0, H / 2],
                               [0, f, W / 2],
                               [0, 0,   1  ]])
    true_points_3d = torch.cat([
            torch.rand((Np, 1))*2,
            torch.rand((Np, 1))*2 + 1.0,
            torch.rand((Np, 1)) + 1.0,], 1)
    true_pose = pp.SE3(torch.tensor([1, 1.5, 0, 0, 0, 0, 1], device=device)).Inv()
    true_points_2d = point2pixel(true_points_3d, intrinsics, true_pose)
    detected_points = true_points_2d + torch.cat([
            (torch.rand((Np, 1))-0.5)*point_noise,
            (torch.rand((Np, 1))-0.5)*point_noise], 1)
    prior_pose = true_pose * pp.randn_SE3(sigma=pose_noise)

    # move data to device
    intrinsics, true_points_3d, detected_points, prior_pose = intrinsics.to(device), \
            true_points_3d.to(device), detected_points.to(device), prior_pose.to(device)
    inputs = (intrinsics, true_points_3d, detected_points, prior_pose)
    model = PoseEstimation(prior_pose).to(device)

    strategy = pp.optim.strategy.TrustRegion(radius=1e6)
    kernel = (ppok.Scale().to(device), ppok.Huber().to(device))
    weight = (torch.eye(6, device=device), torch.eye(2, device=device))
    optimizer = pp.optim.LM(model, strategy=strategy, kernel=kernel)

    last_loss = float("inf")
    timer = Timer()
    for idx in range(100):
        loss = optimizer.step(inputs, weight=weight)
        print('Pose loss %.7f @ %dit'%(loss, idx))
        if loss < 1e-5 or (last_loss - loss) < 1e-5:
            print('Early Stoping!')
            print('Optimization Early Done with loss:', loss.item())
            break
        last_loss = loss

    torch.testing.assert_close(true_pose, model.get_pose(), atol=1e-2, rtol=1e-2)
    print("Time of optimization : {}".format(timer.toc()))
    print("True pose : \n{}".format(true_pose))
    print("Optimized pose : \n{}".format(model.get_pose()))
