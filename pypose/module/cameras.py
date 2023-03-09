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
        return pypose.Inv(self.pose)[..., :3]

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
    """
    A class which stores a batch of parameters to generate a batch of
    transformation matrices using the multi-view geometry convention for
    perspective camera.
    """

    def __init__(
            self,
            pose: Optional[pypose.LieTensor] = None,
            P: Optional[torch.Tensor] = None,
    ):
        """

        Args:
            pose (LieTensor): A tensor of shape (B, *) where B is the batch size.
            P (Optional): A projection matrix of shape (N, 3, 3)
        """
        super().__init__(pose=pose)
        if P is not None:
            self.P = P
        elif self.P is None:
            self.P = torch.eye(3)

    def projection_transform(self, points):
        """
        Transform camera view coordinate points to pixel locations.
        Args:
            points (torch.Tensor): A tensor of shape (B, N, 3) where B is the batch size.
        Returns:
            points (torch.Tensor): A tensor of shape (B, N, 2) where B is the batch size.
        """
        img_repj = torch.matmul(points, self.P.transpose(-2, -1))
        return img_repj[..., :2] / img_repj[..., 2:]

    def is_perspective(self):
        return True
