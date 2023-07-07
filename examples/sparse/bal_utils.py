"""
This file contains the helper functions for the Bundle Adjustment in the Large dataset.

The dataset is from the following paper:
Sameer Agarwal, Noah Snavely, Steven M. Seitz, and Richard Szeliski.
Bundle adjustment in the large.
In European Conference on Computer Vision (ECCV), 2010.

Link to the dataset: https://grail.cs.washington.edu/projects/bal/
"""

import os, torch
import pypose as pp
import matplotlib.pyplot as plt
from bal_loader import build_pipeline

def reprojerr(pose, points, pixels, intrinsics, distortions, point_index, camera_index):
    """
    Calculates batched per-pixel reprojection error considering camera distortion.
    Using BAL's suggestion:

    We use a pinhole camera model; the parameters we estimate for each camera area rotation R, a translation t, a focal length f and two radial distortion parameters k1 and k2. The formula for projecting a 3D point X into a camera R,t,f,k1,k2 is:
    P  =  R * X + t       (conversion from world to camera coordinates)
    p  = -P / P.z         (perspective division)
    p' =  f * r(p) * p    (conversion to pixel coordinates)
    where P.z is the third (z) coordinate of P. In the last equation, r(p) is a function that computes a scaling factor to undo the radial distortion:
    r(p) = 1.0 + k1 * ||p||^2 + k2 * ||p||^4.

    Args:
        pose (``LieTensor``): The camera extrinsics.
            The shape has to be (..., 7).
        points (``torch.Tensor``): The object points in world coordinate.
            The shape has to be (..., N, 3).
        pixels (``torch.Tensor``): The image points. The associated pixel.
            The shape has to be (..., N, 2).
        intrinsics (``torch.Tensor``): intrinsic matrices.
            The shape has to be (..., 3, 3).
        distortions (``torch.Tensor``): The camera distortions.
            The shape has to be (..., 2).
        point_index (``torch.Tensor``): The indices of points.
            The shape has to be (..., N).
        camera_index (``torch.Tensor``): The indices of cameras.
            The shape has to be (..., N).
    Returns:
        Per-pixel reprojection error. The shape is (..., N).
    """
    # convert to camera coordinates
    points = points[point_index, None]
    pose = pose[camera_index]
    points = pose.unsqueeze(-2) @ points
    points = points.squeeze(-2)

    # perspective division
    points_proj = -points[:, :2] / points[:, -1:]

    # convert to pixel coordinates
    intrinsics = intrinsics[camera_index]
    distortions = distortions[camera_index]
    f = intrinsics[:, 0, 0]
    k1 = distortions[:, 0]
    k2 = distortions[:, 1]
    n = torch.sum(points_proj**2, dim=-1)
    r = 1.0 + k1 * n + k2 * n**2
    img_repj = f[:, None] * r[:, None] * points_proj

    # calculate the reprojection error
    loss = (img_repj - pixels).norm(dim=-1)

    return loss

def reprojerr_native(pose, points, pixels, intrinsics, point_index, camera_index):
    """
    Calculates batched per-pixel reprojection error without considering camera distortion.
    Using native pypose implementation.

    Args:
        pose (``LieTensor``): The camera extrinsics.
            The shape has to be (..., 7).
        points (``torch.Tensor``): The object points in world coordinate.
            The shape has to be (..., N, 3).
        pixels (``torch.Tensor``): The image points. The associated pixel.
            The shape has to be (..., N, 2).
        intrinsics (``torch.Tensor``): intrinsic matrices.
            The shape has to be (..., 3, 3).
        point_index (``torch.Tensor``): The indices of points.
            The shape has to be (..., N).
        camera_index (``torch.Tensor``): The indices of cameras.
            The shape has to be (..., N).
    Returns:
        Per-pixel reprojection error. The shape is (..., N).
    """
    loss = pp.reprojerr(points[point_index, None],
                        pixels[:, None],
                        intrinsics[camera_index],
                        pose[camera_index]).squeeze(-1)
    return loss

def visualize_loss_history(loss_history, path_to_img="loss_history.png", title="Loss history"):
    """
    Visualize the loss history.

    Args:
        loss_history (list): A list of loss values.
        path_to_img (str, optional): Path to save the image. Defaults to "loss_history.png".
    Returns:
        None
    """
    plt.plot(loss_history)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title(title)
    plt.savefig(path_to_img)
    plt.close()
    print(f"Loss history saved to {path_to_img}")

def visualize_loss_per_observation(loss, path_to_img="loss_per_observation.png", title="Loss per observation"):
    """
    Visualize the loss per observation.

    Args:
        loss (np.ndarray): A numpy array of loss values.
        path_to_img (str, optional): Path to save the image. Defaults to "loss_per_observation.png".
    Returns:
        None
    """
    plt.plot(loss)
    plt.xlabel('Pixel Observation')
    plt.ylabel('Loss')
    plt.title(title)
    plt.savefig(path_to_img)
    plt.close()
    print(f"Loss per observation saved to {path_to_img}")


def _test():
    # test reprojerr
    def filter_problem(x):
        return x['problem_name'] == 'problem-49-7776-pre'
    dataset_pipeline = build_pipeline(dataset='ladybug', cache_dir='bal_data')\
        .filter(filter_problem)
    dataset_iterator = iter(dataset_pipeline)
    dataset = next(dataset_iterator)
    loss_l2 = reprojerr(dataset['camera_extrinsics'],
                 dataset['points_3d'],
                 dataset['points_2d'],
                 dataset['camera_intrinsics'],
                 dataset['camera_distortions'],
                 dataset['point_index_of_observations'],
                 dataset['camera_index_of_observations'])
    loss_native_l2 = reprojerr_native(dataset['camera_extrinsics'],
                    dataset['points_3d'],
                    dataset['points_2d'],
                    dataset['camera_intrinsics'],
                    dataset['point_index_of_observations'],
                    dataset['camera_index_of_observations'])
    # native implementation and the implementation in this file should have small l1 difference
    # as their only difference is the camera distortion
    loss_l1 = torch.sqrt(loss_l2)
    loss_native_l1 = torch.sqrt(loss_native_l2)
    assert torch.allclose(loss_l1, loss_native_l1, atol=50), "reprojerr implementation incorrect."
    print("reprojerr implementation ok.")

if __name__ == '__main__':
    _test()
