"""
This file contains the dense bundle adjustment solver for the Bundle Adjustment in the Large dataset.

The dataset is from the following paper:
Sameer Agarwal, Noah Snavely, Steven M. Seitz, and Richard Szeliski.
Bundle adjustment in the large.
In European Conference on Computer Vision (ECCV), 2010.

Link to the dataset: https://grail.cs.washington.edu/projects/bal/
"""

import torch, argparse
import pypose as pp
from bal_loader import build_pipeline
from bal_utils import reprojerr, visualize_loss_history, visualize_loss_per_observation
from tqdm import trange

def bundle_adjustment(dataset: dict, num_opt_steps: int = 1000):
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
    num_opt_steps : int, optional
        Number of optimization steps, by default 1000

    Returns
    -------
    camera_extrinsics : pp.LieTensor (n_cameras, 7)
        The optimized camera extrinsics.
    points_3d : torch.Tensor (n_points, 3)
        The optimized 3D points.
    """
    print(f'Solving {dataset["problem_name"]}')
    print(f'Number of optimization steps: {num_opt_steps}')

    # use pytorch's adam for dense ba, device is automatically set to cuda if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    for k, v in dataset.items():
      if isinstance(v, torch.Tensor):
          dataset[k] = v.to(device)

    dataset['camera_extrinsics'].requires_grad_(True)
    dataset['points_3d'].requires_grad_(True)

    optimizer = torch.optim.Adam([dataset['camera_extrinsics'], dataset['points_3d']], lr=3e-4)
    t = trange(num_opt_steps, desc='Loss', leave=True)
    loss_history = []
    loss_vector_history = []

    for _ in t:
        optimizer.zero_grad()
        loss = reprojerr(dataset['camera_extrinsics'],
                 dataset['points_3d'],
                 dataset['points_2d'],
                 dataset['camera_intrinsics'],
                 dataset['camera_distortions'],
                 dataset['point_index_of_observations'],
                 dataset['camera_index_of_observations'])

        loss_vector_history.append(loss.detach().cpu().numpy())
        loss = torch.mean(loss)
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())
        t.set_description(f'Loss: {loss.item():.4f}')

    print(f'Initial loss: {loss_history[0]}')
    print(f'Final loss: {loss_history[-1]}')
    visualize_loss_history(loss_history)
    visualize_loss_per_observation(loss_vector_history[0], path_to_img="loss_per_observation_initial.png", title="Loss per observation (initial)")
    visualize_loss_per_observation(loss_vector_history[-1], path_to_img="loss_per_observation_final.png", title="Loss per observation (final)")

    return dataset['camera_extrinsics'], dataset['points_3d']


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description="This script runs bundle adjustment on the BAL dataset. To understand what datasets and problems are available, please refer to https://grail.cs.washington.edu/projects/bal",
                                     epilog="Example of optimizing `problem-49-7776-pre` in the `ladybug` dataset:\n python3 ba_dense.py --dataset ladybug --problem problem-49-7776-pre --steps 1000",
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--steps', type=int, default=1000, help='Number of optimization steps.')
    parser.add_argument('--dataset', type=str, default='ladybug', help='BAL dataset name, please refer to https://grail.cs.washington.edu/projects/bal for possible values.')
    parser.add_argument('--problem', type=str, default='problem-49-7776-pre', help='BAL problem name, please refer to https://grail.cs.washington.edu/projects/bal for possible values.')
    args = parser.parse_args()
    # load dataset
    def filter_problem(x):
        return x['problem_name'] == args.problem
    dataset_pipeline = build_pipeline(dataset=args.dataset, cache_dir='bal_data')\
        .filter(filter_problem)
    dataset_iterator = iter(dataset_pipeline)
    # run bundle adjustment
    dataset = next(dataset_iterator)
    _, _ = bundle_adjustment(dataset, args.steps)
