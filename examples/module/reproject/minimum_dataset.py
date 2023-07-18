import cv2
import torch
import pypose as pp
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path


class ReprojErrDataset(Dataset):
    def __init__(self, image_dir: Path, flow_dir: Path, depth_dir: Path, gtmotion_path: Path):
        super().__init__()
        assert image_dir.exists() and flow_dir.exists() and depth_dir.exists() and gtmotion_path.exists()

        images = [f for f in image_dir.iterdir() if f.is_file() and f.suffix == ".png"]
        flows  = [f for f in flow_dir.iterdir() if f.is_file() and f.suffix == ".npy"]
        depths = [f for f in depth_dir.iterdir() if f.is_file() and f.suffix == ".npy"]
        images.sort(); flows.sort(); depths.sort()

        self.gt_motions = self._load_tartanair_gt(gtmotion_path)

        self.load_pairs = [pair for pair in zip(images[:-1], images[1:], flows, depths)]
        self.length = len(self.load_pairs)

    def _load_image(self, f: Path):
        return torch.tensor(cv2.imread(str(f))).permute(2, 0, 1) / 255.

    def _load_flow(self, f: Path):
        return torch.tensor(np.load(str(f))).permute(2, 0, 1)

    def _load_depth(self, f: Path):
        return torch.tensor(np.load(str(f))).unsqueeze(dim=0)

    def _load_tartanair_gt(self, path: Path):
        poses = pp.SE3(np.loadtxt(str(path)))
        num_pose = poses.size()[0]
        motion_list = []
        for idx in range(num_pose - 1):
            curr_pose, next_pose = poses[idx], poses[idx + 1]
            motion_list.append(pp.Inv(curr_pose) * next_pose)
        return motion_list

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
        image1 = self._load_image(image1_name)   # Image - [3, 480, 640]
        image2 = self._load_image(image2_name)   # Image - [3, 480, 640]
        flow   = self._load_flow(flow_name)      # Flow  - [2, 480, 640]
        depth  = self._load_depth(depth_name)    # Depth - [1, 480, 640]

        pts1 = self._select_points(image1)
        pts2 = self._match_points(pts1, flow)
        pts1_z = depth[0, pts1[0], pts1[1]]

        return image1, image2, pts1_z, pts1, pts2, self.gt_motions[index]








if __name__ == "__main__":
    ROOT = "/data/TartanAirSample"
    data = ReprojErrDataset(Path(ROOT, "image_left"), Path(ROOT, "flow"), Path(ROOT, "depth_left"))

    i1, i2, f, d, p1, p2 = data[0]
    print(i1.shape, i2.shape, f.shape, d.shape, p1[0].shape, p1[1].shape, p2[0].shape, p2[1].shape)
