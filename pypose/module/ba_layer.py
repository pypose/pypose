
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

    def project_3d_2_2d_normalized(self, x):
        '''
        x (Tensor): [b, 3].
        '''
        raise NotImplementedError()

    def get_projection_matrix(self):
        '''
        Return the 3x3 projection matrix.
        '''
        raise NotImplementedError()

class PinholeCamera(CameraModel):
    def __init__(self, 
        fx:float, 
        fy:float, 
        cx:float, 
        cy:float, 
        shape:List[int], # [height, width]
        eps:float=0.0,
        device='cuda'):

        super().__init__()

        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.shape = shape

        self.eps = eps # Used for dealing with x[:, 2] = 0.

        self.device = device

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

    def get_projection_matrix(self, requires_grad=False):
        K = torch.eye(3, dtype=torch.float, device=self.device, requires_grad=requires_grad)
        K[0, 0] = self.fx
        K[1, 1] = self.fy
        K[0, 2] = self.cx
        K[1, 2] = self.cy
        return K

# Constant camera intrinsics.
class BALayer(nn.Module):
    def __init__(self, 
        n_cam_poses, 
        n_points3d, 
        track_normalizer:float=None # Should be positive if not None.
        ):

        super().__init__()

        # Camera pose represented by SE3.
        self.cam_poses = pp.identity_SE3(n_cam_poses)

        # Tracked 3D points represented by 3-vectors.
        self.points3d = torch.zeros( (n_points3d, 3), dtype=torch.float )

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

    def forward(self, proj_mats, feats, feat_img, feat_loc, tracks):
        '''
        F is number of feature maps, a.k.a, images.
        N is number of features in all feature maps.
        M is number of matcheds between two features.

        proj_mats (Tesnor): [B, 3, 3], projection matrices for all the cameras.
        feats (Tensor): [B, C, H, W], 2D feature maps statcked along the B channel. 
        feat_img (Tensor): [N, 1], dtype=torch.int64, feature-feature map/image/camera association.
        feat_loc (Tensor): [N, 2], dtype=torch.float, 2D feature locations inside feature map indicated by feat_img.
        tracks (Tensor): [M, 2], dtype=torch.int64, list of matched features. Lower feature indices come first.
        '''

        assert ( self.cameras is not None ), \
            f'self.cameras is None. '
        assert ( len(self.cameras) == self.cam_poses.shape[0] ), \
            f'len(self.cameras) = {len(self.cameras)}, self.cam_poses.shape = {self.cam_poses.shape}'

        # Build disjoint sets from tracks.

        # Test if we have enough camera poses and tracks.
        # Allocate more when necessary.

        # 

    def find_3d_point_association(self, N, tracks):
        '''
        N (int): Number of features.
        tracks (Tensor): [M, 2], dtype=torch.int64, list of matched features. Lower feature indices come first.
        '''

        assert ( N > 1 )

        device = tracks.device
        M = tracks.shape[0]

        # Build the adjacency matrix of all the visual features.
        diag_indices = torch.arange(N, device=device)
        row_indices = torch.cat( 
            ( tracks[0, :], tracks[1, :], diag_indices ) )
        col_indices = torch.cat(
            ( tracks[1, :], tracks[0, :], diag_indices ) )
        values = torch.ones( ( M + M + N, ), dtype=torch.float, device=device )
        
        A = torch.sparse_coo_tensor( 
            (row_indices, col_indices), values, dtype=torch.float, device=device )
        
        # Populate all possible edges between the features.
        # A loop?
        # N?
        T = A
        for _ in range(N-1):
            T = torch.sparse.mm(A, T)

        # Get all the upper-triangle indices of T.
        indices_T = T.coalesce().indices()
        upper = indices_T[0] <= indices_T[1]
        upper_indices_T = indices_T[:, upper]

        # Build the mapping between 3D points and 2D features.
        # A loop?

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