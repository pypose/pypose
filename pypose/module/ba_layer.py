
# System.
from typing import List
import math
# from networkx.utils import UnionFind

# PyTorch.
import torch
from torch import is_inference, nn

# functorch
from functorch import vmap

# pypose.
import pypose as pp

# Move this function to a dedicated shared package later.
def sparse_eye(n, dtype=None, device=None, requires_grad=False):
    '''
    Get a nxn sparse diagnal matrix with coo layout.

    n (integer): dimension of the sparse matrix.

    Returns:
    A coo sparse matrix.
    '''

    assert isinstance(n, int), f'n must be an integer, n = {n}, type(n) = {type(n)}'
    assert n > 0, f'n must > 0, n = {n}'

    indices = torch.arange(0, n, dtype=torch.int, device=device)
    values  = torch.ones((n, ), dtype=dtype, device=device)
    return torch.sparse_coo_tensor( 
        indices.repeat(2, 1), values, (n, n), 
        dtype=dtype, device=device, requires_grad=requires_grad )

# Move this function to a dedicated shared package later.
def nearest_log2(v):
    '''
    This function finds the nearest integer power of 2 that is
    smaller than v.

    v (numeric)

    Returns
    Integer
    '''

    return int( math.floor( math.log2(v) ) )

# Move this function to a dedicated shared package later.
def separate_power2(v):
    '''
    This function breaks an integer v into two integers a and b
    such that v = a + b and a is an integer power of 2.
    '''

    assert v >= 2, f'v must >= 2, v = {v}'

    n = nearest_log2( v / 2 + 1 )
    a = int(2**n)
    return a, v - a

# Move this function to a dedicated shared package later.
def sparse_matrix_power(A, ep):
    '''
    Recursvie function. 
    
    If ep == 0, return a sparse identity matrix.
    If ep == 1, return A.
    If ep == 2, return A^{2}.
    If ep > 2, then call itself on two separate powers.

    Note, ep == 0 is not handled.

    A (sparse Tensor)
    ep (integer): the number of powers.

    Returns
    A^{ep}
    '''
    assert A.is_sparse, \
        f'A must be sparse matrix Tensor. '

    assert isinstance(ep, int), \
        f'ep should be integer type. '

    assert ep >= 0, \
        f'ep must >= 0, ep = {ep}'

    if ep == 0:
        return sparse_eye(A.shape[0], dtype=A.dtype, device=A.device, requires_grad=A.requires_grad)

    if ep == 1:
        return A

    if ep == 2:
        return torch.sparse.mm(A, A)

    # Find the separate power values.
    ea, eb = separate_power2(ep)

    if eb == 0:
        return sparse_matrix_power(A, ea)
    else:
        return torch.sparse.mm( 
            sparse_matrix_power(A, ea),
            sparse_matrix_power(A, eb) )

# Move this function to a dedicated shared package later.
def int_matrix_power(A, ep):
    '''
    Raise matrix A to an integer power of ep.
    ep < 0 is not handled.

    A (Tensor): matrix represented as Tensor. Could be sparse.
    ep (int): the power.

    Returns:
    A^{ep}
    '''

    assert A.shape[0] == A.shape[1], \
        f'A must be a square matrix. A.shape = {A.shape}. '

    assert ep >= 0, \
        f'ep must >= 0, ep = {ep}'

    # Check if A is sparse.
    if not A.is_sparse:
        return torch.linalg.matrix_power(A, ep)

    # ========== A is a sparse matrix. ==========

    # Do the power computation.
    return sparse_matrix_power(A, ep)
    
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

def find_leading_feature_from_full_matches(index, full_matches=None):
    '''
    This function returns the index of the leading feature. The current feature is 
    referred to by index. The leading feature and the current feature are referring
    to the same 3D point.

    The leading feature is the feature that has the smallest index among all the features
    that are associated with the same 3D point.

    The leading feature could be the current feature. In that case, the index of the 
    current feature will be returned.

    full_matches contains all the possible matches among all the features. It is assumed that
    the value in the first row of full_matches is always smaller than the second row.

    full_matches has a default value of None to force PyTorch vmap to NOT vectorize it.
    When call this function, full_matches cannot be None.

    index (Tensor): shape of (1, ), the index of the current feature.
    full_maches (Tensor): 2xM feature matching Tensor.

    Returns:
    The index of the leading feature.
    '''

    # assert index.dtype == torch.int64, \
    #     f'index must be a tensor with dtype == torch.int64, index.dtype = {index.dtype}'

    # The following does not work because nonzeros is not supported here.
    # Find the occurance of the current feature index.
    # This casuses host-device synchronization if full_maches is on CUDA. See the documents.
    # f = ( full_matches[1, :] == index ).nonzero(as_tuple=True)[0]

    # The following doese not work because of data-dependent control flow
    # if f.nelement() == 0:
    #     # Not present in the second row of full_maches. 
    #     # This means it is the leading feature.
    #     return index

    mask = full_matches[1, :] == index
    mask = mask.type(torch.int)

    # Since we assume that the smallest matching index is the index of the leading feature.
    buffered_indices = mask * full_matches[0, :] + (1 - mask) * index

    return torch.min(buffered_indices)

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

        proj_mats (Tesnor): [F, 3, 3], projection matrices for all the cameras.
        feats (Tensor): [F, C, H, W], 2D feature maps statcked along the B channel. 
        feat_img (Tensor): [N, 1], dtype=torch.int64, feature-feature map/image/camera association.
        feat_loc (Tensor): [N, 2], dtype=torch.float, 2D feature locations inside feature map indicated by feat_img.
        tracks (Tensor): [M, 2], dtype=torch.int64, list of matched features. Lower feature indices come first.
        '''

        assert ( self.cameras is not None ), \
            f'self.cameras is None. '
        assert ( len(self.cameras) == self.cam_poses.shape[0] ), \
            f'len(self.cameras) = {len(self.cameras)}, self.cam_poses.shape = {self.cam_poses.shape}'

        F = feats.shape[0]
        N = feat_img.shape[0]

        # Find the association between features and 3D points.
        association = self.find_3d_point_association( F, N, tracks )

        # Test if we have enough camera poses and tracks.
        # Allocate more when necessary.

        

    def associate_3d_point_indices_2_features(self, N, full_matches):
        '''
        full_mactches is a 2xM Tensor. Where M is the maximum number of matches among
        all the N features. full_matches must contain all the possible matches.

        This function generates a new tensor of lenght N, where every element shows the
        index of the 3D point this element is associated with.

        N (integer): number of total features.
        full_matches (Tensor).

        Returns:
        A length N tensor and the number of total 3D points.
        '''

        # Create a Tensor that stores feature indices that referring to themselves.
        feature_indices = torch.arange(0, N, dtype=torch.int64, device=full_matches.device)

        # Vectorization.
        leading_indices = vmap(find_leading_feature_from_full_matches)(feature_indices, full_matches=full_matches)

        # Assign 3D point indices to the leading features.
        association = torch.zeros((N, ), dtype=torch.int64, device=full_matches.device)
        self_leading_indices = feature_indices[ leading_indices == feature_indices ]
        association[ self_leading_indices ] = \
            torch.arange( 0, self_leading_indices.nelement(), dtype=torch.int64, device=full_matches.device )

        # Assign 3D point indices be referencing to the leading feature indices.
        non_self_leading_indices = feature_indices[ leading_indices != feature_indices ]
        association[ non_self_leading_indices ] = association[ leading_indices[ non_self_leading_indices ] ]

        return association

    def find_3d_point_association(self, n_img, N, tracks):
        '''
        n_img (int): Number of images.
        N (int): Number of features.
        tracks (Tensor): [M, 2], dtype=torch.int64, list of matched features. Lower feature indices come first.
        '''

        assert ( n_img > 1 ), f'n_img must > 1, n_img = {n_img}'
        assert ( N > 1 ), f'N must > 1, N = {N}'

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
        T = int_matrix_power(A, n_img)

        # Get all the upper-triangle indices of T.
        indices_T = T.coalesce().indices()
        upper = indices_T[0] <= indices_T[1]
        upper_indices_T = indices_T[:, upper]

        # Build the mapping between 3D points and 2D features.
        association = self.associate_3d_point_indices_2_features( N, upper_indices_T )

        return association

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