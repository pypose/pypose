
# System.
from typing import List
from networkx.utils import UnionFind

# PyTorch.
import torch
from torch import nn

# pypose.
import pypose as pp

class CameraModel(object):
    def __init__(self):
        super().__init__()

    def project_3d_2_2d(self, x):
        '''
        x (Tensor): [b, 3].
        '''
        raise NotImplementedError()

class PinholeCamera(CameraModel):
    def __init__(self, 
        fx:float, 
        fy:float, 
        cx:float, 
        cy:float, 
        shape:List[int], # [height, width]
        eps:float=0.0):

        super().__init__()

        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.shape = shape

        self.eps = eps # Used for dealing with x[:, 2] = 0.

    def project_3d_2_2d(self, x):
        '''
        x (Tensor): [b, 3].
        '''
        assert ( x.shape[-1] == 3 ), \
            f'x has wrong dimention. x.shape = {x.shape}'

        p = torch.zeros( (x.shape[0], 2), dtype=x.dtype, device=x.device )
        # TODO: We do not care about x[:, 2] being zero, do we?
        p[:, 0] = self.fx * x[:, 0] / ( x[:, 2] + self.eps ) + self.cx
        p[:, 1] = self.fy * x[:, 1] / ( x[:, 2] + self.eps ) + self.cy
        return p

    def project_3d_2_2d_normalized(self, x):
        '''
        x (Tensor): [b, 3].
        '''
        p = self.project_3d_2_2d(x)
        p[:, 0] = p[:, 0] / ( self.shape[1] - 1 )
        p[:, 1] = p[:, 1] / ( self.shape[0] - 1 )

        return p

# Constant camera intrinsics.
class BALayer(nn.Module):
    def __init__(self, 
        n_cam_poses, 
        n_points3d, 
        cameras:List[CameraModel]=None,
        track_normalizer:float=None # Should be positive if not None.
        ):

        super().__init__()

        # Camera pose represented by SE3.
        self.cam_poses = pp.identity_SE3(n_cam_poses)

        # Tracked 3D points represented by 3-vectors.
        self.points3d = torch.zeros( (n_points3d, 3), dtype=torch.float )

        if ( cameras is not None ):
            assert ( len(cameras) == n_cam_poses ), \
                f'len(cameras) = {len(cameras)}, n_cam_poses = {n_cam_poses}'
        self.cameras = cameras

        if ( track_normalizer is not None ):
            assert ( track_normalizer > 0 ), \
                f'track_normalizer = {track_normalizer}'
        self.track_normalizer = track_normalizer

    def allocate_if_not_enough(self, n_cam_poses, n_points3d):
        n_cam_poses_ori = self.cam_poses.gshape[0]

        if ( n_cam_poses > n_cam_poses_ori ):
            cam_poses = pp.identity_SE3(n_cam_poses, dtype=self.cam_poses.dtype, device=self.cam_poses.device)
            cam_poses[ :n_cam_poses_ori, ... ] = self.cam_poses
            self.cam_poses = cam_poses

        n_points3d_ori = self.points3d.shape[0]
        if ( n_points3d > n_points3d_ori ):
            tracks = torch.zeros( (n_points3d, 3), dtype=self.points3d.dtype, device=self.points3d.device )
            tracks[ :n_points3d_ori, ... ] = self.points3d
            self.points3d = tracks

    def forward(self, feats, feat_img, feat_loc, tracks):
        '''
        F is number of feature maps, a.k.a, images.
        N is number of features in all feature maps.
        M is number of matcheds between two features.

        feats (Tensor): [B, F, C, H, W], 2D feature maps statcked along the F channel. Should it be [B, C, F, H, W]?
        feat_img (Tensor): [B, N, 1], dtype=torch.int64, feature-feature map/image/camera association.
        feat_loc (Tensor): [B, N, 2], dtype=torch.float, 2D feature locations inside feature map indicated by feat_img.
        tracks (Tensor): [B, M, 2], dtype=torch.int64, list of matched features. Lower feature indices come first.
        '''

        assert ( self.cameras is not None ), \
            f'self.cameras is None. '
        assert ( len(self.cameras) == self.cam_poses.shape[0] ), \
            f'len(self.cameras) = {len(self.cameras)}, self.cam_poses.shape = {self.cam_poses.shape}'

        # Build disjoint sets from tracks.

        # Test if we have enough camera poses and tracks.
        # Allocate more when necessary.

        # 

    def normalize_feat_loc(self, feat_loc):
        '''
        Normalize feat_loc.

        feat_loc (Tensor): [B, N, 2]

        Returns:
        The normalized feat locations in the image/feature map.
        '''
        raise NotImplementedError()

    def normalize_positions(self):
        '''
        Scale self.cam_poses such that the positions are normalized.
        '''
        raise NotImplementedError()

    def denormalize_positions(self):
        '''
        Apply the inverse of normalization_positions().
        '''
        raise NotImplementedError()

    def residual(self, x_obsv, x_proj):
        '''
        x_obsv: observed feature location, [2, 1].
        x_proj: re-projected (3D) points, [2, 1].

        Returns:
        Residuals, [2, 1].
        '''
        raise NotImplementedError()

    def cost(self, r):
        '''
        Take a residual vector, compute the scalar cost value.

        r: residuals, [2, 1].

        Returns:
        Cost value. Scalar.
        '''
        raise NotImplementedError()