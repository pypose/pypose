import torch
import argparse
import pypose as pp
from pathlib import Path
import pypose.optim as ppopt
import matplotlib.pyplot as plt
from dataset import ReprojErrDataset, visualize, report_pose_error


class ReprojectErrorGraph(torch.nn.Module):
    def __init__(self, K, pts1, pts2, depth, init_T) -> None:
        super().__init__()
        self.register_buffer("K", K)
        self.register_buffer("pts1_v", pts1[0])
        self.register_buffer("pts1_u", pts1[1])
        self.register_buffer("pts2_v", pts2[0])
        self.register_buffer("pts2_u", pts2[1])
        self.fx, self.fy = K[0, 0], K[1, 1]
        self.cx, self.cy = K[0, 2], K[1, 2]

        self.T = pp.Parameter(init_T)
        self.depth = torch.nn.Parameter(depth)

    def pts3d(self) -> torch.Tensor:
        pts3d_z = self.depth
        pts3d_x = ((self.pts1_u - self.cx) * pts3d_z) / self.fx
        pts3d_y = ((self.pts1_v - self.cy) * pts3d_z) / self.fy
        return torch.stack([pts3d_x, pts3d_y, pts3d_z], dim=1)  # Nx3

    @torch.no_grad()
    def reproject(self) -> torch.Tensor:
        pts3d = self.pts3d()
        reproj_uv = pp.function.point2pixel(pts3d, self.K, self.T.Inv())
        return torch.roll(reproj_uv, shifts=1, dims=1)

    @torch.no_grad()
    def error(self) -> float:
        pts2 = torch.stack([self.pts2_u, self.pts2_v], dim=1)
        err_uv = pp.function.reprojerr(self.pts3d(), pts2, self.K, self.T.Inv(), reduction='none')
        return torch.mean(torch.norm(err_uv, dim=1, p=2)).item()

    def forward(self) -> torch.Tensor:
        pts2 = torch.stack([self.pts2_u, self.pts2_v], dim=1)
        err_uv = pp.function.reprojerr(self.pts3d(), pts2, self.K, self.T.Inv(), reduction='none')
        return err_uv


if __name__ == "__main__":
    plt.ion()
    parser = argparse.ArgumentParser(
        description="Estimate trajectory by minimize reprojection error in adjacent frames")
    parser.add_argument("--device", action="store", default="cuda",
                        help="Device to run optimization (cuda / cpu)")
    parser.add_argument("--save", action="store", default="est_traj.txt",
                        help="Path to save the estimated trajectory")
    parser.add_argument("--dataroot", action="store", default="/data/TartanAirSample_Compressed",
                        help="Root directory for the dataset")
    parser.add_argument("--vectorize", action="store_true", default=False,
        help="Add this flag to use 8-point-algorithm for Essential matrix instead of 5-point algorithm")
    args = parser.parse_args()
    dataroot = Path(args.dataroot)
    device, vectorize = args.device, args.vectorize
    K = torch.tensor([[320., 0., 320.], [0., 320., 240.], [0., 0., 1.]])

    dataset = ReprojErrDataset(dataroot=dataroot)

    for img1, img2, depth, pts1, pts2, gt_motion in dataset:
        # Noisy initial pose and depth  noise ~ N(avg=0, std=.1)
        init_T = (gt_motion * pp.randn_SE3(sigma=.1)).to(device)
        depth = depth + torch.randn_like(depth) * .1

        print('Initial Motion Error:')
        report_pose_error(init_T, gt_motion.to(device))

        graph = ReprojectErrorGraph(K, pts1, pts2, depth, init_T).to(device)
        kernel = ppopt.kernel.Huber(delta=0.1)
        corrector = ppopt.corrector.FastTriggs(kernel)
        optimizer = ppopt.LM(
            graph,
            solver=ppopt.solver.Cholesky(),
            strategy=ppopt.strategy.TrustRegion(radius=1e3),
            kernel=kernel, corrector=corrector,
            min=1e-8, vectorize=vectorize, reject=128,
        )
        scheduler = ppopt.scheduler.StopOnPlateau(optimizer, steps=25, patience=4, decreasing=1e-6, verbose=True)

        # Optimize Reproject Pose Graph Optimization ##########################
        print('\tInitial graph error:', graph.error())
        while scheduler.continual():
            visualize(img1, img2, pts1, pts2, graph, scheduler.steps)
            loss = optimizer.step(input=())
            scheduler.step(loss)
        print('\tFinal graph error:', graph.error())
        #######################################################################

        final_T = pp.SE3(graph.T.data.detach())

        print('Optimized Motion Error')
        report_pose_error(final_T, gt_motion.to(device))
        print("\n\n")
