import cv2
import torch
import argparse
import numpy  as np
import pypose as pp
import matplotlib as mpl

from pathlib import Path
import pypose.optim as ppopt
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from minimum_dataset import ReprojErrDataset

class ReprojectErrorModel(torch.nn.Module):
    def __init__(self, K, pts1, pts2, depth, est_T12) -> None:
        super().__init__()
        self.register_buffer("K", K)
        self.register_buffer("pts1_v", pts1[0])
        self.register_buffer("pts1_u", pts1[1])
        self.register_buffer("pts2_v", pts2[0])
        self.register_buffer("pts2_u", pts2[1])
        self.fx, self.fy = K[0, 0], K[1, 1]
        self.cx, self.cy = K[0, 2], K[1, 2]

        self.reproj2d = None
        self.T_12 = pp.Parameter(est_T12)
        self.depth = torch.nn.Parameter(depth)

    @property
    def N(self): return self.pts1_v.shape[0]

    def reproject(self) -> None:
        pts3d_z = self.depth
        pts3d_x = ((self.pts1_u - self.cx) * pts3d_z) / self.fx
        pts3d_y = ((self.pts1_v - self.cy) * pts3d_z) / self.fy
        pts3d = torch.stack([pts3d_x, pts3d_y, pts3d_z], dim=1)  # Nx3

        reproj_uv = self.K @ (self.T_12.Inv().Act(pts3d)).T
        reproj_uv = reproj_uv[:2] / reproj_uv[2]
        reproj_vu = torch.roll(reproj_uv, shifts=1, dims=0)

        self.reproj2d = reproj_vu.T

    @torch.no_grad()
    def error(self) -> float:
        self.reproject()
        err_vu = self.reproj2d - torch.stack([self.pts2_v, self.pts2_u], dim=1)
        return torch.mean(torch.norm(err_vu, dim=1, p=2)).item()

    def forward(self) -> torch.Tensor:
        self.reproject()
        err_vu = self.reproj2d - torch.stack([self.pts2_v, self.pts2_u], dim=1)
        return err_vu.contiguous()

class ReprojectionErrorBackend:
    def __init__(self,
                 K: torch.Tensor,
                 device="cuda",
                 vectorize: bool=True) -> None:
        self.K      = K
        self.device = device
        self.vectorize = vectorize

        self.NED2CV = pp.from_matrix(
            torch.tensor([[0., 1., 0., 0.],
                          [0., 0., 1., 0.],
                          [1., 0., 0., 0.],
                          [0., 0., 0., 1.]], dtype=torch.float32),
            pp.SE3_type).to(self.device)
        self.CV2NED = self.NED2CV.Inv()

        self.colorbar = mpl.colormaps['coolwarm']
        self.norm     = mpl.colors.Normalize(vmin=0, vmax=1)
        plt.ion()

    def solve(self, image1, image2, depth, pts1, pts2, est_T12) -> pp.SE3:
        target = ReprojectErrorModel(self.K, pts1, pts2, depth, est_T12)
        target.to(self.device)

        solver    = ppopt.solver.Cholesky()
        strategy  = ppopt.strategy.TrustRegion(radius=1e3)
        kernel    = ppopt.kernel.Huber(delta=0.1)
        corrector = ppopt.corrector.FastTriggs(kernel)
        optimizer = ppopt.LM(
            target, solver=solver, strategy=strategy, min=1e-8, vectorize=self.vectorize, reject=128,
            kernel=kernel, corrector=corrector
        )
        scheduler = ppopt.scheduler.StopOnPlateau(optimizer, steps=25, patience=4, decreasing=1e-6, verbose=True)

        print("\tInitial_error:", target.error())
        pts2_vu        = torch.stack(pts2, dim=1)
        pts1_v, pts1_u = pts1[0].int(), pts1[1].int()

        while scheduler.continual():
            reproj_vu  = target.reproj2d.detach().cpu()
            reproj_err = torch.norm(pts2_vu - reproj_vu, dim=1).detach().cpu().numpy()
            reproj_vu  = reproj_vu.int()

            display_img0 = np.array(image1.permute((1, 2, 0)).clone().cpu().numpy() * 255, dtype=np.uint8)
            display_img0 = cv2.cvtColor(display_img0, cv2.COLOR_BGR2GRAY)
            display_img0 = np.stack([display_img0] * 3, axis=2)

            display_img1 = np.array(image2.permute((1, 2, 0)).clone().cpu().numpy() * 255, dtype=np.uint8)
            display_img1 = cv2.cvtColor(display_img1, cv2.COLOR_BGR2GRAY)
            display_img1 = np.stack([display_img1] * 3, axis=2)

            display_img  = np.concatenate([display_img0, display_img1], axis=1)

            plt.clf()
            plt.axis('off')
            plt.imshow(display_img, interpolation='nearest')
            for idx in range(target.N):
                err = reproj_err[idx].item()
                v1      , u1       = pts1_v[idx].item()  , pts1_u[idx].item()
                reproj_v, reproj_u = reproj_vu[idx, 0].item(), reproj_vu[idx, 1].item()
                plt.plot([u1, reproj_u + display_img0.shape[1]], [v1, reproj_v], color=self.colorbar(err))
            plt.title(f"Step: {scheduler.steps}, Error: {target.error()}")
            divider = make_axes_locatable(plt.gca())
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(mpl.cm.ScalarMappable(norm=self.norm, cmap=self.colorbar), cax=cax)
            plt.pause(0.1)

            loss = optimizer.step(input=())
            scheduler.step(loss)

        print("\tFinal_error:", target.error())
        return pp.SE3(target.T_12.data.detach())


Parser = argparse.ArgumentParser(description="Estimate trajectory by minimize reprojection error in adjacent frames")
Parser.add_argument(
    "--device", action="store", default="cuda",
    help="Device to run optimization (cuda / cpu)"
)
Parser.add_argument(
    "--save", action="store", default="est_traj.txt",
    help="Path to save the estimated trajectory"
)
Parser.add_argument(
    "--dataroot", action="store", default="/data/TartanAirSample",
    help="Root directory for the dataset"
)
Parser.add_argument(
    "--vectorize", action="store_true", default=False,
    help="Add this flag to use 8-point-algorithm for Essential matrix instead of 5-point algorithm"
)

if __name__ == "__main__":
    Args = Parser.parse_args()
    ROOT = Path(Args.dataroot)
    SAVE = Path(Args.save)
    DEVICE = Args.device
    VECTORIZE = Args.vectorize

    PT_PER_FRAME = 100
    K = torch.tensor([
        [320., 0.  , 320.],
        [0.  , 320., 240.],
        [0.  , 0.  , 1.  ]
    ])

    dataset = ReprojErrDataset(
        Path(ROOT, "image_left"),
        Path(ROOT, "flow"),
        Path(ROOT, "depth_left"),
        Path(ROOT, "pose_left.txt")
    )
    backend = ReprojectionErrorBackend(K, device=DEVICE, vectorize=VECTORIZE)

    for image1, image2, depth, pts1, pts2, gt_motion in dataset:
        gt_motion = backend.NED2CV @ gt_motion.to(DEVICE) @ backend.CV2NED

        initial_T12 = gt_motion * pp.randn_SE3(sigma=.2).to(DEVICE)
        initial_err_rot = (initial_T12.Inv() * gt_motion).rotation().Log().norm(dim=-1).item()
        initial_err_trans = (initial_T12.Inv() * gt_motion).translation().norm(dim=-1).item()
        print(f"Initial guess error: Rot - {round(initial_err_rot, 4)} | Trans - {round(initial_err_trans, 4)}")

        final_T12 = backend.solve(image1, image2, depth, pts1, pts2, initial_T12)

        final_err_rot = (final_T12.Inv() * gt_motion).rotation().Log().norm(dim=-1).item()
        final_err_trans = (final_T12.Inv() * gt_motion).translation().norm(dim=-1).item()
        print(f"Final error: Rot - {round(final_err_rot, 4)} | Trans - {round(final_err_trans, 4)}")
        print("\n\n")

