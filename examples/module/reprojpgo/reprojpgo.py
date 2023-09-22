import torch
import argparse
import pypose as pp
from torch import nn
from pathlib import Path
from pypose.optim import LM
from pypose.optim.kernel import Huber
from pypose.optim.solver import Cholesky
from pypose.optim.strategy import TrustRegion
from pypose.optim.corrector import FastTriggs
from pypose.optim.scheduler import StopOnPlateau
from pypose.function.geometry import pixel2point, reprojerr
from dataset import MiniTartanAir, visualize, report_pose_error


class LocalBundleAdjustment(nn.Module):
    def __init__(self, K, pts1, pts2, depth, init_T) -> None:
        super().__init__()
        self.register_buffer("K", K)
        self.register_buffer("pts1", pts1)  # N x 2, uv coordinate
        self.register_buffer("pts2", pts2)  # N x 2, uv coordinate

        self.T = pp.Parameter(init_T)
        self.depth = nn.Parameter(depth)

    def forward(self) -> torch.Tensor:
        pts3d = pixel2point(self.pts1, self.depth, self.K)
        return reprojerr(pts3d, self.pts2, self.K, self.T.Inv(), reduction='none')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Estimate camera motion by optimizing reprojerr graph "
                    "between adjacent frames")
    parser.add_argument("--dataroot", action="store",
                        default="./examples/module/reprojpgo/save/",
                        help="Root directory for the dataset")
    parser.add_argument("--device", action="store", default="cuda",
                        help="Device to run optimization (cuda / cpu)")
    parser.add_argument("--vectorize", action="store_true", default=True,
                        help="Vectorize when optimizing reprojerr graph.")
    parser.add_argument("--dnoise", default=0.1, type=float,
                        help="Noise level of the depth")
    parser.add_argument("--pnoise", default=0.1, type=float,
                        help="Noise level of the pose")
    args = parser.parse_args()
    dataroot = Path(args.dataroot)
    device, vectorize = args.device, args.vectorize
    K = torch.tensor([[320., 0., 320.], [0., 320., 240.], [0., 0., 1.]])

    dataset = MiniTartanAir(dataroot=dataroot)

    for img1, img2, depth, pts1, pts2, gt_motion in dataset:
        # Noisy initial pose and depth noise ~ N(avg=0, std=args.pnoise)
        init_T = (gt_motion * pp.randn_SE3(sigma=args.pnoise)).to(device)
        depth = depth + torch.randn_like(depth) * args.dnoise

        print('Initial Motion Error:')
        report_pose_error(init_T, gt_motion.to(device))

        graph = LocalBundleAdjustment(K, pts1, pts2, depth, init_T).to(device)
        kernel = Huber(delta=0.1)
        corrector = FastTriggs(kernel)
        optimizer = LM(graph, solver=Cholesky(),
                              strategy=TrustRegion(radius=1e3),
                              kernel=kernel,
                              corrector=corrector,
                              min=1e-8,
                              reject=128,
                              vectorize=vectorize)
        scheduler = StopOnPlateau(optimizer, steps=25,
                                             patience=4,
                                             decreasing=1e-6,
                                             verbose=True)

        # Optimize Reproject Pose Graph Optimization ###########################
        while scheduler.continual():
            loss = optimizer.step(input=())
            visualize(img1, img2, pts1, pts2, graph, loss, scheduler.steps)
            scheduler.step(loss)
        ########################################################################

        optimized_T = pp.SE3(graph.T.data.detach())

        print('Optimized Motion Error')
        report_pose_error(optimized_T, gt_motion.to(device))
        print("\n\n")
