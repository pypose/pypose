from typing import Optional

import pypose
import torch.nn


class CamerasBase(torch.nn.Module):
    """
    `CamerasBase` implements a base class for all cameras.
    For cameras, there are two different coordinate systems (or spaces)
    - World coordinate system: This is the system the object lives - the world.
    - Camera view coordinate system: This is the system that has its origin on
        the camera and the Z-axis perpendicular to the image plane.
        We assume that +X points left, and +Y points up and
        +Z points out from the image plane.
        The transformation from world --> view happens after applying a rotation (R)
        and translation (T)
    CameraBase defines methods that are common to all camera models:
        - `camera_center` that returns the optical center of the camera in
            world coordinates
        - `world_to_view_transform` which returns a 3D transform from
            world coordinates to the camera view coordinates (R, T)
        - `full_projection_transform` which composes the projection
            transform (P) with the world-to-view transform (R, T)

    Args:
        pose (LieTensor): A tensor of shape (B, *) where B is the batch size.
    """

    def __init__(self, pose=None, ):
        super().__init__()
        if pose is not None:
            self.pose = pose
        elif self.pose is None:
            self.pose = pypose.identity_SE3()

    def get_camera_center(self):
        """
        Returns the camera center in world coordinates.
        Returns:
            A tensor of shape (B, 3) where B is the batch size.
        """
        return self.pose.Inv()[..., :3]

    def world_to_view_transform(self, points):
        """
        Take world coordinate points and transform them to camera view coordinate.
        Args:
            points (torch.Tensor): A tensor of shape (B, N, 3) where B is the batch size.
        Returns:
            points (torch.Tensor): A tensor of shape (B, N, 3) where B is the batch size.
        """
        return pypose.Act(self.pose.unsqueeze(-2), points)

    def full_projection_transform(self, points):
        """
        Composes the projection transform (P) with the world-to-view transform (R, T)
        Args:
            points (torch.Tensor): A tensor of shape (B, N, 3) where B is the batch size.
        Returns:
            points (torch.Tensor): A tensor of shape (B, N, 2) where B is the batch size.
        """
        return self.projection_transform(self.world_to_view_transform(points))

    def projection_transform(self, points):
        """
        Transform camera view coordinate points to pixel locations.
        Args:
            points (torch.Tensor): A tensor of shape (B, N, 3) where B is the batch size.
        Returns:
            points (torch.Tensor): A tensor of shape (B, N, 2) where B is the batch size.
        """
        raise NotImplementedError

    def reprojection_error(self, pts_w, img_pts):
        """
        Calculate the reprojection error.
        Args:
            pts_w: The object points in world coordinate. The shape is (..., N, 3).
            img_pts: The image points. The shape is (..., N, 2).
        Returns:
            error: The reprojection error. The shape is (..., ).
        """
        # Calculate the image points
        img_repj = self.full_projection_transform(pts_w)

        error = torch.linalg.norm(img_repj - img_pts, dim=-1)
        error = torch.mean(error, dim=-1)

        return error


class PerspectiveCameras(CamerasBase):
    r"""
    A class which stores a batch of parameters to generate a batch of
    transformation matrices using the multi-view geometry convention for
    perspective camera.

    Args:
        pose (LieTensor): A tensor of shape (B, *) where B is the batch size.
        intrinsics (Optional): A projection matrix of shape (N, 3, 3)

    Examples:
        >>> import torch
        >>> import pypose
        >>> # create some random data
        >>> pose = pypose.SE3([ 0.0000, -8.0000,  0.0000,  0.0000, -0.3827,  0.0000,  0.9239])
        >>> f = 2
        >>> img_size = (7, 7)
        >>> projection_matrix = torch.tensor([[f, 0, img_size[0] / 2,], [0, f, img_size[1] / 2,], [0, 0, 1, ]])
        >>> pts_w = torch.tensor([[ 2.8284,  8.0000,  0.0000], [ 2.1213,  8.0000,  0.7071], [ 0.7071,  9.0000,  0.7071], [ 0.7071,  8.0000,  0.7071], [ 5.6569, 13.0000, -1.4142]])
        >>> # instantiate the camera
        >>> camera = pypose.module.PerspectiveCameras(pose=pose, intrinsics=projection_matrix)
        >>> # transform the points to image coordinates
        >>> img_pts = camera.full_projection_transform(pts_w)
        >>> img_pts
        tensor([[5.4998, 3.5000],
                [4.4999, 3.5000],
                [3.4999, 5.5000],
                [3.4999, 3.5000],
                [6.8329, 6.8330]])
    """

    def __init__(
            self,
            pose: Optional[pypose.LieTensor] = None,
            intrinsics: Optional[torch.Tensor] = None,
    ):
        super().__init__(pose=pose)
        if intrinsics is not None:
            self.intrinsics = intrinsics
        elif self.intrinsics is None:
            self.intrinsics = torch.eye(3)

    def projection_transform(self, points):
        r"""
        Transform camera view coordinate points to pixel locations.
        Args:
            points (torch.Tensor): A tensor of shape (B, N, 3) where B is the batch size.
        Returns:
            points (torch.Tensor): A tensor of shape (B, N, 2) where B is the batch size.
        """
        img_repj = points.matmul(self.intrinsics.transpose(-2, -1))
        return img_repj[..., :2] / img_repj[..., 2:]

    def is_perspective(self):
        return True
