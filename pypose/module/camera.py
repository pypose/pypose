import torch
import pypose as pp
from torch import nn
from typing import Optional


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
        - `world2camera` which returns a 3D transform from
            world coordinates to the camera view coordinates (R, T)
        - `world2pixel` which composes the projection
            transform (P) with the world-to-view transform (R, T)

    Args:
        pose (LieTensor): A tensor of shape (B, *) where B is the batch size.
    """

    def __init__(self, pose=None, ):
        super().__init__()
        if pose is not None:
            self.pose = pp.Parameter(pose)
        elif self.pose is None:
            self.pose = pp.Parameter(pp.identity_SE3())

    def get_camera_center(self):
        """
        Returns the camera center in world coordinates.
        Returns:
            A tensor of shape (B, 3) where B is the batch size.
        """
        return self.pose.Inv().translation()

    def world2camera(self, points):
        """
        a.k.a world to view transform.
        Take world coordinate points and transform them to camera view coordinate.
        Args:
            points (torch.Tensor): A tensor of shape (B, N, 3) where B is the batch size.
        Returns:
            points (torch.Tensor): A tensor of shape (B, N, 3) where B is the batch size.
        """
        return self.pose.unsqueeze(-2) @ points

    def world2pixel(self, points):
        """
        a.k.a full projection transform.
        Composes the projection transform (P) with the world-to-view transform (R, T)
        Args:
            points (torch.Tensor): A tensor of shape (B, N, 3) where B is the batch size.
        Returns:
            points (torch.Tensor): A tensor of shape (B, N, 2) where B is the batch size.
        """
        return self.camera2pixel(self.world2camera(points))

    def camera2pixel(self, points):
        """
        a.k.a projection transform.
        Transform camera view coordinate points to pixel locations.
        Args:
            points (torch.Tensor): A tensor of shape (B, N, 3) where B is the batch size.
        Returns:
            points (torch.Tensor): A tensor of shape (B, N, 2) where B is the batch size.
        """
        raise NotImplementedError

    def reprojection_error(self, points, pixels):
        """
        Calculate the reprojection error.
        Args:
            points: The object points in world coordinate. The shape is (..., N, 3).
            pixels: The image points. The shape is (..., N, 2).
        Returns:
            error: The reprojection error. The shape is (..., ).
        """
        # Calculate the image points
        img_repj = self.world2pixel(points)

        error = torch.linalg.norm(img_repj - pixels, dim=-1)
        error = error.mean(dim=-1)

        return error


class Camera(CamerasBase):
    r"""
    A class which stores a batch of parameters to generate a batch of
    transformation matrices using the multi-view geometry convention for
    perspective camera.

    Args:
        pose (LieTensor): A tensor of shape (B, *) where B is the batch size.
        intrinsics (Optional): A projection matrix of shape (N, 3, 3)

    Examples:
        >>> import torch
        >>> import pypose as pp
        >>> # create some random data
        >>> pose = pp.SE3([ 0.0000, -8.0000,  0.0000,  0.0000, -0.3827,  0.0000,  0.9239])
        >>> f = 2
        >>> img_size = (7, 7)
        >>> projection = torch.tensor([[f, 0, img_size[0] / 2,], [0, f, img_size[1] / 2,], [0, 0, 1, ]])
        >>> pts_w = torch.tensor([[ 2.8284,  8.0000,  0.0000],
        ...                       [ 2.1213,  8.0000,  0.7071],
        ...                       [ 0.7071,  9.0000,  0.7071],
        ...                       [ 0.7071,  8.0000,  0.7071],
        ...                       [ 5.6569, 13.0000, -1.4142]])
        >>> # instantiate the camera
        >>> camera = pp.module.Camera(pose=pose, intrinsics=projection)
        >>> # transform the points to image coordinates
        >>> img_pts = camera.world2pixel(pts_w)
        >>> img_pts
        tensor([[5.4998, 3.5000],
                [4.4999, 3.5000],
                [3.4999, 5.5000],
                [3.4999, 3.5000],
                [6.8329, 6.8330]], grad_fn=<DivBackward0>)
    """

    def __init__(
            self,
            pose: Optional[pp.LieTensor] = None,
            intrinsics: Optional[torch.Tensor] = None,
    ):
        super().__init__(pose=pose)
        if intrinsics is not None:
            self.intrinsics = nn.Parameter(intrinsics)
        elif self.intrinsics is None:
            self.intrinsics = nn.Parameter(torch.eye(3))

    def camera2pixel(self, points):
        r"""
        Transform camera view coordinate points to pixel locations.
        Args:
            points (torch.Tensor): A tensor of shape (B, N, 3) where B is the batch size.
        Returns:
            points (torch.Tensor): A tensor of shape (B, N, 2) where B is the batch size.
        """
        # this is equivalent to left multiplying the intrinsics to the points
        img_repj = points.matmul(self.intrinsics.mT)
        return img_repj[..., :2] / img_repj[..., 2:]
