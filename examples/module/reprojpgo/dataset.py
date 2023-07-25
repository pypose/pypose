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


class ReprojErrDataset(Dataset):
    def __init__(self, dataroot: Path):
        super().__init__()
        self.NED2CV = pp.from_matrix(torch.tensor(
            [[0., 1., 0., 0.],
             [0., 0., 1., 0.],
             [1., 0., 0., 0.],
             [0., 0., 0., 1.]], dtype=torch.float32),
            pp.SE3_type)
        self.CV2NED = self.NED2CV.Inv()

        data_source = torch.load(dataroot)
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
        flow = self.flows[index].to(torch.float32)    # flow size [2, 480, 640]
        depth = self.depths[index].to(torch.float32)  # depth size [1, 480, 640]
        gt_motion = self.NED2CV @ self.gt_motions[index] @ self.CV2NED

        pts1 = select_points(image1)
        pts2 = match_points(pts1, flow)
        pts1_z = depth[0, pts1[..., 1], pts1[..., 0]]
        image1 = visualize_image(image1)
        image2 = visualize_image(image2)

        return image1, image2, pts1_z, pts1, pts2, gt_motion


def select_points(image: torch.Tensor):
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
    selected_points = image_grad > image_grad_avg + 3. * image_grad_std
    border_mask = torch.zeros_like(selected_points)
    border_mask[..., 5:-5, 5:-5] = 1.

    result_points = selected_points * border_mask
    indices = torch.nonzero(result_points, as_tuple=False)

    # Randomly select points, only take 100 points
    perm = torch.randperm(indices.shape[0])[:100]
    # .roll(...): vu -> uv coordinate
    pts_uv = indices[perm][..., 1:].roll(shifts=1, dims=[1])
    return pts_uv


def match_points(pts1: torch.Tensor, flow: torch.Tensor):
    delta_uv = flow[..., pts1[..., 1], pts1[..., 0]]
    return pts1 + delta_uv.T


def visualize_image(img: torch.Tensor):
    display_img = img.permute((1, 2, 0)).cpu().numpy() * 255
    display_img = display_img.astype(np.uint8)
    display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2GRAY)
    display_img = np.stack([display_img] * 3, axis=2)
    return display_img


def visualize(img1, img2, pts1, pts2, target, step):
    plt.ion()
    color_map = mpl.colormaps['coolwarm']
    color_normalizer = mpl.colors.Normalize(vmin=0, vmax=1)
    display_img = np.concatenate([img1, img2], axis=1)
    reproj_uv = target.reproject().detach().cpu()
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
    plt.title(f"Step: {step}, Error: {target.error()}")
    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(ScalarMappable(norm=color_normalizer, cmap=color_map), cax=cax)
    plt.pause(0.1)


def report_pose_error(curr_pose: pp.SE3, gt_pose: pp.SE3):
    _err = (curr_pose.Inv() * gt_pose)
    _err_rot = _err.rotation().Log().norm(dim=-1).item() * (180 / np.pi)
    _err_trans = _err.translation().norm(dim=-1).item()
    _err_rot = round(_err_rot, 4)
    _err_trans = round(_err_trans, 4)
    print(f"Err Rot (deg) - {_err_rot} | Err Trans (m) - {_err_trans}")
