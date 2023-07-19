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
    def __init__(self, data_root: Path, image_dir: Path, flow_dir: Path, depth_dir: Path, gtmotion_path: Path):
        super().__init__()
        assert image_dir.exists() and flow_dir.exists() and depth_dir.exists() and gtmotion_path.exists()
        self.NED2CV = pp.from_matrix(torch.tensor(
            [[0., 1., 0., 0.],
             [0., 0., 1., 0.],
             [1., 0., 0., 0.],
             [0., 0., 0., 1.]], dtype=torch.float32),
            pp.SE3_type)
        self.CV2NED = self.NED2CV.Inv()

        images = [f for f in image_dir.iterdir() if f.is_file() and f.suffix == ".png"]
        flows = [f for f in flow_dir.iterdir() if f.is_file() and f.suffix == ".npz"]
        depths = [f for f in depth_dir.iterdir() if f.is_file() and f.suffix == ".npz"]
        images.sort()
        flows.sort()
        depths.sort()

        self.gt_motions = self._load_tartanair_gt(gtmotion_path)

        self.load_pairs = [pair for pair in zip(images[:-1], images[1:], flows, depths)]
        self.length = len(self.load_pairs)

    def _load_image(self, f: Path):
        return torch.tensor(cv2.imread(str(f))).permute(2, 0, 1) / 255.

    def _load_flow(self, f: Path):
        return torch.tensor(np.load(str(f)).get("arr_0").astype(np.float32)).permute(2, 0, 1)

    def _load_depth(self, f: Path):
        return torch.tensor(np.load(str(f)).get("arr_0").astype(np.float32)).unsqueeze(dim=0)

    def _visualize_image(self, img: torch.Tensor):
        display_img = np.array(img.permute((1, 2, 0)).clone().cpu().numpy() * 255, dtype=np.uint8)
        display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2GRAY)
        display_img = np.stack([display_img] * 3, axis=2)
        return display_img

    def _load_tartanair_gt(self, path: Path):
        poses = pp.SE3(np.loadtxt(str(path)))
        motions = poses[:-1].Inv() * poses[1:]
        return motions

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
        indices = indices[perm]

        pts_v, pts_u = indices[:, 1], indices[:, 2]
        return pts_v, pts_u

    def _match_points(self, pts1, flow: torch.Tensor):
        pts1_v, pts1_u = pts1
        dv = flow[1, pts1_v, pts1_u]
        du = flow[0, pts1_v, pts1_u]
        return pts1_v + dv, pts1_u + du

    def __len__(self): return self.length

    def __getitem__(self, index):
        image1_name, image2_name, flow_name, depth_name = self.load_pairs[index]
        image1 = self._load_image(image1_name)  # image - shape [3, 480, 640]
        image2 = self._load_image(image2_name)  # image - shape [3, 480, 640]
        flow = self._load_flow(flow_name)  # flow  - shape [2, 480, 640]
        depth = self._load_depth(depth_name)  # depth - shape [1, 480, 640]

        pts1 = self._select_points(image1)
        pts2 = self._match_points(pts1, flow)
        pts1_z = depth[0, pts1[0], pts1[1]]

        display_img1 = self._visualize_image(image1)
        display_img2 = self._visualize_image(image2)
        gt_motion = self.NED2CV @ self.gt_motions[index] @ self.CV2NED
        return display_img1, display_img2, pts1_z, pts1, pts2, gt_motion


def visualize(img1, img2, pts1, pts2, target, step):
    color_map = mpl.colormaps['coolwarm']
    color_normalizer = mpl.colors.Normalize(vmin=0, vmax=1)
    pts1_v, pts1_u = pts1[0].int(), pts1[1].int()
    display_img = np.concatenate([img1, img2], axis=1)
    reproj_vu = target.reproject().detach().cpu()
    reproj_err = torch.norm(torch.stack(pts2, dim=1) - reproj_vu, dim=1).detach().cpu().numpy()
    reproj_vu = reproj_vu.int()

    plt.clf()
    plt.axis('off')
    plt.imshow(display_img, interpolation='nearest')
    for idx in range(target.N):
        err = reproj_err[idx].item()
        v1, u1 = pts1_v[idx].item(), pts1_u[idx].item()
        reproj_v, reproj_u = reproj_vu[idx, 0].item(), reproj_vu[idx, 1].item()
        plt.plot([u1, reproj_u + img1.shape[1]], [v1, reproj_v], color=color_map(err))
    plt.title(f"Step: {step}, Error: {target.error()}")
    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(mpl.cm.ScalarMappable(norm=color_normalizer, cmap=color_map), cax=cax)
    plt.pause(0.1)


def report_pose_error(curr_pose: pp.SE3, gt_pose: pp.SE3):
    initial_err_rot = (curr_pose.Inv() * gt_pose).rotation().Log().norm(dim=-1).item() * (180 / np.pi)
    initial_err_trans = (curr_pose.Inv() * gt_pose).translation().norm(dim=-1).item()
    print(f"Err Rot (deg) - {round(initial_err_rot, 4)} | Err Trans (m) - {round(initial_err_trans, 4)}")