import cv2
import torch
import numpy as np
import pypose as pp
import matplotlib as mpl
from pathlib import Path
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
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

        data_source = torch.load(Path(dataroot, "reproj_dataset.pth"))
        self.images = data_source["images"]
        self.flows: torch.Tensor = data_source["flows"]
        self.depths = data_source["depths"]
        self.gt_motions = data_source["gt_traj"]
        self.length = self.flows.size(0)

    def _visualize_image(self, img: torch.Tensor):
        display_img = np.array(img.permute((1, 2, 0)).clone().cpu().numpy() * 255, dtype=np.uint8)
        display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2GRAY)
        display_img = np.stack([display_img] * 3, axis=2)
        return display_img

    def _select_points(self, image: torch.Tensor):
        image_grad = torch.nn.functional.conv2d(
            image.unsqueeze(dim=0),
            torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]]).float().expand((1, 3, 3, 3)),
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

    def _match_points(self, pts1_uv, flow: torch.Tensor):
        delta_uv = flow[..., pts1_uv[..., 1], pts1_uv[..., 0]]
        return pts1_uv + delta_uv.T

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        image1 = self.images[index]     # image - shape [3, 480, 640]
        image2 = self.images[index + 1] # image - shape [3, 480, 640]
        flow = self.flows[index]        # flow  - shape [2, 480, 640]
        depth = self.depths[index]      # depth - shape [1, 480, 640]
        gt_motion = self.NED2CV @ self.gt_motions[index] @ self.CV2NED

        pts1 = self._select_points(image1)
        pts2 = self._match_points(pts1, flow)
        pts1_z = depth[0, pts1[..., 1], pts1[..., 0]]

        display_img1 = self._visualize_image(image1)
        display_img2 = self._visualize_image(image2)
        return display_img1, display_img2, pts1_z, pts1, pts2, gt_motion


def visualize(img1, img2, pts1, pts2, target, step):
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
        plt.plot([u1, reproj_u + img1.shape[1]], [v1, reproj_v], color=color_map(err))
    plt.title(f"Step: {step}, Error: {target.error()}")
    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(mpl.cm.ScalarMappable(norm=color_normalizer, cmap=color_map), cax=cax)
    plt.pause(0.1)


def report_pose_error(curr_pose: pp.SE3, gt_pose: pp.SE3):
    _err = (curr_pose.Inv() * gt_pose)
    _err_rot = _err.rotation().Log().norm(dim=-1).item() * (180 / np.pi)
    _err_trans = _err.translation().norm(dim=-1).item()
    print(f"Err Rot (deg) - {round(_err_rot, 4)} | Err Trans (m) - {round(_err_trans, 4)}")
