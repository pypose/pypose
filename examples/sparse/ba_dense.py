import torch, argparse, os
import pypose as pp
from bal_loader import build_pipeline


def f(pose, points, pixels, intrinsics, point_index, camera_index):
    loss = pp.reprojerr(points[point_index, None],
                        pixels[:, None],
                        intrinsics[camera_index],
                        pose[camera_index])
    return loss


def bundle_adjustment(dataset: dict):
    """
    Bundle adjustment for the BAL dataset.

    Parameters
    ----------
    dataset : dict
        A dictionary containing the following fields:
        - problem_name: str
            The name of the problem.
        - camera_extrinsics: pp.LieTensor (n_cameras, 7)
            The camera extrinsics.
            First three columns are translation, last four columns is unit quaternion.
        - camera_intrinsics: torch.Tensor (n_cameras, 3, 3)
            The camera intrinsics. Each camera is represented as a 3x3 K matrix.
        - camera_distortions: torch.Tensor (n_cameras, 2)
            The camera distortions. k1 and k2.
        - points_3d: torch.Tensor (n_points, 3)
            contains initial estimates of point coordinates in the world frame.
        - points_2d: torch.Tensor (n_observations, 2)
            contains measured 2-D coordinates of points projected on images in each observations.
        - camera_index_of_observations: torch.Tensor (n_observations,)
            contains indices of cameras (from 0 to n_cameras - 1) involved in each observation.
        - point_index_of_observations: torch.Tensor (n_observations,)
            contains indices of points (from 0 to n_points - 1) involved in each observation.
    Returns
    -------
    ...
    """
    print(f'Solving {dataset["problem_name"]}')

    # use pytorch's adam for dense ba, device is automatically set to cuda if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    for k, v in dataset.items():
      if isinstance(v, torch.Tensor):
          dataset[k] = v.to(device)


    dataset['camera_extrinsics'].requires_grad_(True)
    dataset['points_3d'].requires_grad_(True)

    optimizer = torch.optim.Adam([dataset['camera_extrinsics'], dataset['points_3d']], lr=1e-2)

    for i in range(100):
        optimizer.zero_grad()
        loss = f(dataset['camera_extrinsics'],
                 dataset['points_3d'],
                 dataset['points_2d'],
                 dataset['camera_intrinsics'],
                 dataset['point_index_of_observations'],
                 dataset['camera_index_of_observations'])
        loss = torch.mean(loss)
        loss.backward()
        optimizer.step()
        print(f'Iter {i+1} loss: {loss.item()}')

    return dataset['camera_extrinsics'], dataset['points_3d']


if __name__ == '__main__':
    dataset_pipeline = build_pipeline(dataset='ladybug', cache_dir='bal_data')\
        .filter(lambda x: x['problem_name'] == 'problem-49-7776-pre')
    dataset_iterator = iter(dataset_pipeline)
    dataset = next(dataset_iterator)
    bundle_adjustment(dataset)
