import cv2
import torch
import numpy as np
import pypose as pp
import matplotlib as mpl
from pathlib import Path
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torchvision.datasets.utils import download_and_extract_archive


class MiniTartanAir(Dataset):
    link = 'https://github.com/pypose/pypose/releases/download/v0.5.0/MiniTartanAir.pt.zip'
    def __init__(self, dataroot: Path, download = True):
        super().__init__()
        if download:
            download_and_extract_archive(self.link, str(dataroot))
        self.NED2CV = pp.from_matrix(torch.tensor(
            [[0., 1., 0., 0.],
             [0., 0., 1., 0.],
             [1., 0., 0., 0.],
             [0., 0., 0., 1.]], dtype=torch.float32),
            pp.SE3_type)
        self.CV2NED = self.NED2CV.Inv()

        data_source = torch.load(Path(dataroot, "MiniTartanAir.pt"))
        self.images = data_source["images"]
        self.flows: torch.Tensor = data_source["flows"]
        self.depths = data_source["depths"]
        self.gt_motions = data_source["gt_traj"]
        self.length = self.flows.size(0)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        image1 = self.images[index] / 255.            # image size [3, 480, 640]
        image2 = self.images[index + 1] / 255.        # image size [3, 480, 640]
        flow = self.flows[index].to(torch.float32)    # flow size  [2, 480, 640]
        depth = self.depths[index].to(torch.float32)  # depth size [1, 480, 640]
        gt_motion = self.NED2CV @ self.gt_motions[index] @ self.CV2NED

        pts1 = self.select_points(image1)
        pts2 = self.match_points(pts1, flow)
        pts1_z = depth[0, pts1[..., 1], pts1[..., 0]]

        H, W = image1.size(-2), image1.size(-1) # mask out points outside images
        mask = torch.logical_and(pts2[..., 1] < H, pts2[..., 0] < W)
        pts1, pts2, pts1_z = pts1[mask], pts2[mask], pts1_z[mask]

        return image1, image2, pts1_z, pts1, pts2, gt_motion

    @staticmethod
    def select_points(image: torch.Tensor, num_point: int = 100):
        image_grad = torch.nn.functional.conv2d(
            image.unsqueeze(dim=0),
            torch.tensor(
                [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
            ).float().expand((1, 3, 3, 3)),
            padding=1
        )[0].abs()
        image_grad_avg = image_grad.mean(dim=(1, 2), keepdim=True)
        image_grad_std = image_grad.std(dim=(1, 2), keepdim=True)
        # Positions with sufficient gradient (feature) > +3std
        points = image_grad > image_grad_avg + 3. * image_grad_std
        border_mask = torch.zeros_like(points)
        border_mask[..., 5:-5, 5:-5] = 1.

        points = points * border_mask
        selected_points = torch.nonzero(points, as_tuple=False)

        # Randomly select points
        perm = torch.randperm(selected_points.shape[0])[:num_point]
        # vu -> uv coordinate
        pts_uv = selected_points[perm][..., 1:].roll(shifts=1, dims=[1])
        return pts_uv

    @staticmethod
    def match_points(pts1: torch.Tensor, flow: torch.Tensor):
        return pts1 + flow[..., pts1[..., 1], pts1[..., 0]].T


def visualize_image(img: torch.Tensor):
    display_img = img.permute((1, 2, 0)).cpu().numpy() * 255
    display_img = display_img.astype(np.uint8)
    display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2GRAY)
    display_img = np.stack([display_img] * 3, axis=2)
    return display_img


def visualize(img1, img2, pts1, pts2, target, loss, step):
    plt.ion()

    img1 = visualize_image(img1)    # Convert to black&white OpenCV displayable
    img2 = visualize_image(img2)    # format

    color_map = mpl.colormaps['coolwarm']
    color_normalizer = mpl.colors.Normalize(vmin=0, vmax=1)
    display_img = np.concatenate([img1, img2], axis=1)

    pts3d = pp.function.geometry.pixel2point(target.pts1, target.depth, target.K)
    reproj_uv = pp.function.point2pixel(pts3d, target.K, target.T.Inv())
    reproj_uv = reproj_uv.detach().cpu()

    reproj_err = torch.norm(pts2 - reproj_uv, dim=1).detach().cpu().numpy()

    plt.clf()
    plt.axis('off')
    plt.imshow(display_img, interpolation='nearest')
    for idx in range(target.pts1.size(0)):
        err = reproj_err[idx].item()
        u1, v1 = pts1[idx, 0].item(), pts1[idx, 1].item()
        reproj_u, reproj_v = reproj_uv[idx, 0].item(), reproj_uv[idx, 1].item()
        plt.plot(
            [u1, reproj_u + img1.shape[1]],
            [v1, reproj_v]
            , color=color_map(err)
        )
    plt.title(f"Step: {step}, ReprojErr: {round(reproj_err.mean().item(), 3)}, "
              f"Residual: {round(loss.item(), 3)}")
    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(ScalarMappable(norm=color_normalizer, cmap=color_map), cax=cax)
    plt.pause(0.1)


def report_pose_error(curr_pose: pp.SE3, gt_pose: pp.SE3):
    _err = (curr_pose.Inv() * gt_pose)
    _err_rot = _err.rotation().Log().norm(dim=-1).item() * (180 / np.pi)
    _err_trans = _err.translation().norm(dim=-1).item()
    print(f"Err Rot (deg): {round(_err_rot, 4)} | Err Trans (m): {round(_err_trans, 4)}")
