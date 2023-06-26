import pypose as pp
from bal_loader import build_pipeline
import torch

def bundle_adjustment(dataset: dict):
    """
    Bundle adjustment for the BAL dataset.

    Parameters
    ----------
    dataset : dict
        A dictionary containing the following fields:
        - problem_name: str
            The name of the problem.
        - camera_extrinsics: pp.LieTensor (n_observation, 7)
            The camera extrinsics, represented as pp.LieTensor, SE3 type
            First three columns are translation, last four columns is unit quaternion.
        - camera_intrinsics: torch.Tensor (n_observation, 3, 3)
            The camera intrinsics. Each camera is represented as a 3x3 K matrix.
        - points_3d: torch.Tensor (n_observation, 3)
            Contains initial estimates of point coordinates in the world frame.
        - points_2d: torch.Tensor (n_observations, 2)
            Contains measured 2-D coordinates of points projected on images in each observations.

    Returns
    -------
    camera_extrinsics : pp.LieTensor (n_observation, 7)
        The camera extrinsics, represented as pp.LieTensor, SE3 type
        First three columns are translation, last four columns is unit quaternion.
    points_3d : torch.Tensor (n_observation, 3)
        Contains optimized estimates of point coordinates in the world frame.
    """
    # use pytorch's adam for dense ba, device is automatically set to cuda if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    for k, v in dataset.items():
        if isinstance(v, torch.Tensor):
            dataset[k] = v.to(device)

    dataset['points_3d'].requires_grad_()
    dataset['camera_extrinsics'].requires_grad_()

    optimizer = torch.optim.Adam([dataset['camera_extrinsics'], dataset['points_3d']], lr=1e-2)
    for i in range(100):
        optimizer.zero_grad()
        loss = pp.reprojerr(dataset['points_3d'],
                            dataset['points_2d'],
                            dataset['camera_intrinsics'],
                            dataset['camera_extrinsics'])
        loss = torch.sum(loss)
        loss.backward()
        optimizer.step()
        print(f'Iter {i+1} loss: {loss.item()}')
    return dataset['camera_extrinsics'], dataset['points_3d']


if __name__ == '__main__':
    dataset_pipeline = build_pipeline(dataset='ladybug', cache_dir='bal_data')
    dataset_iterator = iter(dataset_pipeline)
    dataset = next(dataset_iterator)
    print(f'Solving {dataset["problem_name"]}')
    bundle_adjustment(dataset)
