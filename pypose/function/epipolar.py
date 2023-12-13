import torch
from ransac import RANSAC

def normalize_points(coordinates:torch.Tensor):
    r"""  
    Calculate the normalized transformation of each coordinate set. After the transformation, 
    the centroid of each coordinate set is located at the coordinate origin, 
    and the average distance from the origin is sqrt(2).
    Args:
        coordinates (``torch.Tensor``): Homogeneous coordinates with the shape (..., N, 3).

    Returns:
        (``norm_pts: torch.Tensor,transform: torch.Tensor``):
        ``norm_pts``: normalized coordinates with the shape (..., N, 3).

        ``transform``: transformation matrix with the shape (..., 3, 3).
    
    Example:
        >>> pts = torch.tensor([[ 40, 368],[ 63, 224], [ 85, 447], [108, 151], [ 77, 319], [411, 371],[335, 183],[ 42, 151],[144, 440],
                            [407, 403],[208, 153], [216, 288], [156, 318]])
        >>> PH = pp.cart2homo(pts)
        >>> norm_pts, transfer_matrix = normalize_points(PH)
        >>> norm_pts
        tensor([[-1.2624,  0.6896,  1.0000],
                [-1.0494, -0.6440,  1.0000],
                [-0.8456,  1.4213,  1.0000],
                [-0.6326, -1.3201,  1.0000],
                [-0.9197,  0.2358,  1.0000],
                [ 2.1736,  0.7174,  1.0000],
                [ 1.4697, -1.0238,  1.0000],
                [-1.2439, -1.3201,  1.0000],
                [-0.2992,  1.3565,  1.0000],
                [ 2.1366,  1.0138,  1.0000],
                [ 0.2935, -1.3016,  1.0000],
                [ 0.3676, -0.0513,  1.0000],
                [-0.1881,  0.2266,  1.0000]])
        >>> transfer_matrix
        tensor([[ 0.0093,  0.0000, -1.6329],
                [ 0.0000,  0.0093, -2.7186],
                [ 0.0000,  0.0000,  1.0000]])

    """
    assert coordinates.shape[-1] == 3, "the coordinates has to be the homogeneous!"
    
    # keep data at same device
    device = coordinates.device
    coordinates = coordinates.type(torch.FloatTensor).to(device)  

    # mean = torch.mean(coordinates, dim = 0)
    mean = torch.mean(coordinates, dim = -2, keepdim = True)  # (..., 1, 3)
    scale = 1.4142135623730950488 / (torch.linalg.norm(coordinates - mean, dim = -1).mean(dim = -1) + 1e-8)  # (...)
    mean_x, mean_y, _ = torch.chunk(mean, dim = -1, chunks = 3)
    
    scale, mean_x, mean_y = scale.view(-1,1), mean_x.view(-1,1), mean_y.view(-1,1)
    zeros, ones = torch.zeros_like(scale), torch.ones_like(scale)

    transform = torch.cat((scale, zeros, - scale * mean_x,
                        zeros, scale, - scale * mean_y,
                        zeros, zeros, ones), dim = -1).view(-1,3,3)
    
    if len(coordinates.shape) == 2: transform = transform.squeeze()

    #norm_pts = transform @ coordinates.T
    norm_pts = coordinates @ transform.transpose(-2,-1)  # (..., N, 3)

    return norm_pts, transform

def triangulate_points(coordinates1:torch.Tensor,coordinates2:torch.Tensor,intrinsic1,intrinsic2,R,t):
    r"""
    Convert 2D corresponding coordinates to 3D points.
    
    Args:
        coordinates1 (``torch.Tensor``): Image coordinates with the shape (..., N, 2).
        coordinates2 (``torch.Tensor``): Image coordinates with the shape (..., N, 2).
        intrinsic1 (``torch.Tensor``): The intrinsic matrix of the first camera with the shape (3, 3).
        intrinsic2 (``torch.Tensor``): The intrinsic matrix of the second camera with the shape (3, 3).
        R (``torch.Tensor``): The rotation matrix with the shape(3, 3). 
        t (``torch.Tensor``): The translation matrix with the shape(3, 1). 
    
    Returns:
        (``points3D: torch.Tensor``): 
        ``points3D``: The 3D points with the shape(N, 3). 

    Example:
        >>> K = torch.tensor([[320., 0., 320.], [0., 320., 240.], [0., 0., 1.]])
        >>> R = torch.tensor([[ 0.9997,  0.0097,  0.0229],
        ...                   [-0.0101,  0.9998,  0.0171],
        ...                   [-0.0228, -0.0173,  0.9996]])
        >>> t = torch.tensor([[-0.8864],
        ...                   [-0.1004],
        ...                   [-0.4519]])
        >>> pts1 = torch.tensor([[ 40, 368],[ 63, 224], [ 85, 447], [108, 151], [ 77, 319], [411, 371],[335, 183],[ 42, 151],[144, 440],
        ...                      [407, 403],[208, 153], [216, 288], [156, 318], [104, 319], [ 72, 287], [373, 180], [383, 200], [184, 306],
        ...                      [362, 185], [211, 463], [385, 211], [ 80,  81],[ 78, 195], [398, 223]])
        >>> pts2 = torch.tensor([[ 16.0156, 377.1875],[ 45.9062, 229.5703],[ 59.3750, 459.8750], [ 94.4766, 155.7422], [ 56.5469, 326.3945], [409.6777, 378.8906],
        ...                      [327.7773, 185.5117], [ 27.0625, 156.4453], [121.6250, 452.5000], [402.8477, 412.9609], [196.8281, 156.6094], [202.9375, 293.7734],
        ...                      [139.6562, 324.9727],[ 84.5312, 326.2695], [ 52.7656, 293.5781],[367.6094, 182.1094], [379.1543, 202.5859], [169.1406, 312.4688],
        ...                      [356.0312, 187.2891],[191.6719, 476.9531],[381.6484, 213.8477],[ 91.6328,  92.5391],[ 62.7266, 200.2344],[396.4111, 226.1543]])   
        >>> pts1 = pp.cart2homo(pts1)
        >>> pts2 = pp.cart2homo(pts2)
        >>> pts_3D = triangulate_points(pts1,pts2,K,K,R,t)
        >>> pts_3D
        tensor([[ -10.0953,    4.6160,   11.5376],
                [ -11.0966,   -0.6914,   13.8168],
                [  -8.0963,    7.1338,   11.0252],
                [ -10.4402,   -4.3847,   15.7593],
                [  -9.5682,    3.1112,   12.6001],
                [   6.1911,    8.9134,   21.7702],
                [   0.9383,   -3.5658,   20.0089],
                [ -12.6423,   -4.0489,   14.5526],
                [  -6.4165,    7.2937,   11.6670],
                [   4.6667,    8.7445,   17.1632],
                [  -6.1782,   -4.8013,   17.6530],
                [  -5.1073,    2.3574,   15.7148],
                [  -7.2632,    3.4550,   14.1722],
                [  -8.7290,    3.1932,   12.9319],
                [ -10.0989,    1.9141,   13.0308],
                [   3.5632,   -4.0352,   21.5107],
                [   4.6211,   -2.9355,   23.4701],
                [  -6.2971,    3.0563,   14.8167],
                [   2.7450,   -3.5959,   20.9111],
                [  -4.1121,    8.4159,   12.0736],
                [   4.8978,   -2.1866,   24.1109],
                [-151.2956, -100.3358,  202.3115],
                [ -11.1175,   -2.0683,   14.7011],
                [   6.6520,   -1.4514,   27.2892]])
    """
    assert coordinates1.shape[-1] == 2, "The shape must be (..., N, 2)!"
    M = torch.cat([intrinsic1,torch.zeros_like(intrinsic1[..., :1])],dim =-1)
    Proj = intrinsic2 @ torch.cat([R,t],dim =-1)
    M = torch.broadcast_to(M,Proj.shape)

    coordinates10, coordinates11 = torch.chunk(coordinates1, dim = -1, chunks = 2)
    coordinates20, coordinates21 = torch.chunk(coordinates2, dim = -1, chunks = 2)

    M0, M1, M2 = torch.chunk(M, dim = -2, chunks = 3)
    Proj_row0, Proj_row1, Proj_row2 = torch.chunk(Proj, dim = -2, chunks = 3)

    # consturct the A and b matrices
    row1 = coordinates10 * M2 - M0               # (..., N, 4)
    row2 = coordinates11 * M2 - M1               # (..., N, 4)
    row3 = coordinates20 * Proj_row2 - Proj_row0     # (..., N, 4)
    row4 = coordinates21 * Proj_row2 - Proj_row1     # (..., N, 4)

    _, row1_0_2, row1_3 = torch.tensor_split(row1,(0,3),dim =-1)
    _, row2_0_2, row2_3 = torch.tensor_split(row2,(0,3),dim =-1)
    _, row3_0_2, row3_3 = torch.tensor_split(row3,(0,3),dim =-1)
    _, row4_0_2, row4_3 = torch.tensor_split(row4,(0,3),dim =-1)

    row1_0_2, row2_0_2, row3_0_2, row4_0_2 = row1_0_2.unsqueeze(-2), row2_0_2.unsqueeze(-2), row3_0_2.unsqueeze(-2), row4_0_2.unsqueeze(-2)  # (..., N, 1, 3)
    row1_3, row2_3, row3_3, row4_3 = row1_3.unsqueeze(-1), row2_3.unsqueeze(-1), row3_3.unsqueeze(-1), row4_3.unsqueeze(-1)  # (..., N, 1, 1)
    
    A  = torch.cat([row1_0_2,row2_0_2,row3_0_2,row4_0_2], dim = -2)   # (..., N, 4, 3)
    AT = A.transpose(-2,-1)                          # (..., N, 3, 4)
    b  = - torch.cat([row1_3,row2_3,row3_3,row4_3], dim = -2)  # (..., N, 4, 1)

    points3D = torch.linalg.inv(AT @ A) @ AT @ b     # (..., N, 3, 1)

    return points3D.squeeze()

def eight_pts_alg(coordinates1:torch.Tensor, coordinates2:torch.Tensor):
    r"""
    A minimum of eight corresponding points from two images 
    to obtain an initial estimate of the essential or fundamental matrix. 

    Args:
        coordinates1 (``torch.Tensor``): Homogeneous coordinates with the shape (..., N, 3).
        coordinates2 (``torch.Tensor``): Homogeneous coordinates with the shape (..., N, 3).
        
    
    Returns:
        (``F: torch.Tensor``):
        ``F``: essential or fundamental matrix with the shape (..., 3, 3).
    
    Example:
        >>> pts1 = torch.tensor([[ 40, 368],[ 63, 224], [ 85, 447], [108, 151], [ 77, 319], [411, 371],[335, 183],[ 42, 151],[144, 440],
        ...                      [407, 403],[208, 153], [216, 288], [156, 318], [104, 319], [ 72, 287], [373, 180], [383, 200], [184, 306],
        ...                      [362, 185], [211, 463], [385, 211], [ 80,  81],[ 78, 195], [398, 223]])
        >>> pts2 = torch.tensor([[ 16.0156, 377.1875],[ 45.9062, 229.5703],[ 59.3750, 459.8750], [ 94.4766, 155.7422], [ 56.5469, 326.3945], [409.6777, 378.8906],
        ...                      [327.7773, 185.5117], [ 27.0625, 156.4453], [121.6250, 452.5000], [402.8477, 412.9609], [196.8281, 156.6094], [202.9375, 293.7734],
        ...                      [139.6562, 324.9727],[ 84.5312, 326.2695], [ 52.7656, 293.5781],[367.6094, 182.1094], [379.1543, 202.5859], [169.1406, 312.4688],
        ...                      [356.0312, 187.2891],[191.6719, 476.9531],[381.6484, 213.8477],[ 91.6328,  92.5391],[ 62.7266, 200.2344],[396.4111, 226.1543]])   
        >>> pts1 = pp.cart2homo(pts1)
        >>> pts2 = pp.cart2homo(pts2)
        >>> F = eight_pts_alg(pts1,pts2)
        >>> F
        tensor([[ 4.6117e-07, -9.0610e-05,  2.7522e-02],
                [ 9.4227e-05,  3.9505e-06, -8.7046e-02],
                [-2.9752e-02,  8.4636e-02,  1.0000e+00]])
    """
    assert coordinates1.shape == coordinates2.shape, "Point sets has to be the same shape!"
    assert coordinates1.shape[-1] == 3, "The shape must be (..., N, 3)!"
    assert coordinates1.shape[-2] > 7, "The number of point pairs must be greater than 7!"

    norm_pts1, transfer_matrix1 = normalize_points(coordinates1)
    norm_pts2, transfer_matrix2 = normalize_points(coordinates2)

    x1, y1, _ = torch.chunk(norm_pts1, dim = -1, chunks = 3)
    x2, y2, _ = torch.chunk(norm_pts2, dim = -1, chunks = 3)

    A = torch.cat([x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, torch.ones_like(x1)], dim=-1)
    _,_,VT = torch.linalg.svd(A.transpose(-2, -1) @ A)

    if len(VT.shape) == 2: M = VT[-1,:].view(3,3)  
    if len(VT.shape) == 3: M = VT[:,-1,:].view(-1,3,3)  # (..., 3, 3)

    # Enforce rank 2 of F, put the last singular value to 0
    U,S,VT = torch.linalg.svd(M)

    if len(S.shape) == 1: S[-1] = 0.0
    if len(S.shape) == 2: S[:,-1] = 0.0  
    M_new = U @ torch.diag_embed(S) @ VT

    F = transfer_matrix2.transpose(-2,-1) @ M_new @ transfer_matrix1

    # Make the value of the last row and column is one.
    if len(F.shape) == 2: last_val = F[-1:,-1:]
    if len(F.shape) == 3: last_val = F[:,-1:,-1:]

    F = torch.where(last_val.abs()> 0.0, F / last_val, F)
    return F

def find_essential_mat(coordinates1:torch.Tensor,coordinates2:torch.Tensor,intrinsic,method = 'none',terminate =10000, threshold = 0.5):
    r""" 
    Comupte the essential matrix.

    Args:
        coordinates1 (``torch.Tensor``): Image coordinates with the shape (N, 2).
        coordinates2 (``torch.Tensor``): Image coordinates with the shape (N, 2).
        intrinsic (``torch.Tensor``): The intrinsic matrix with the shape (3, 3).
        method (``str``, optional): The method to calculate the fundamental matrix.``'none'``| ``'RANSAC'`` | ``'RANSAC_ADAPT'`` 

                              ``'none'``: No method is applied. This method will compute the E or F with all points in the corresponding points set.

                              ``'RANSAC'``: RANSAC method is applied. Use iteration number as termination condition.

                              ``'RANSAC_ADAPT'``: RANSAC method is applied. Use the probabilty of obtaining a success model as termination condition.

        terminate (``int or float``, optional): The maximum number of iterations( type: int, method: 'RANSAC') in RANSAC method.

                                                The probabilty of obtaining a model fitted by all selected data that are inliers ( type: float, method: 'RANSAC_ADAPT') in RANSAC method.

        threshold (``float``, optional): The maximum tolerable distance from the corresponding points to the epipolarlines.
                                         It is only used in RANSAC method. 
    
    Returns:
        (``E: torch.Tensor, mask: torch.Tensor``):
        ``E``: Essential matrix with the shape (3, 3).

        ``mask``: For ``'RANSAC'`` or ``'RANSAC_ADAPT'``, it will return the index of the inliers in the corresponding points set.The shape is (..., 1).

                  For ``'none'``, no mask will be returned.
    
    Example:
        
        >>> K = torch.tensor([[320., 0., 320.], [0., 320., 240.], [0., 0., 1.]])
        >>> pts1 = torch.tensor([[ 40, 368],[ 63, 224], [ 85, 447], [108, 151], [ 77, 319], [411, 371],[335, 183],[ 42, 151],[144, 440],
        ...                      [407, 403],[208, 153], [216, 288], [156, 318], [104, 319], [ 72, 287], [373, 180], [383, 200], [184, 306],
        ...                      [362, 185], [211, 463], [385, 211], [ 80,  81],[ 78, 195], [398, 223]])
        >>> pts2 = torch.tensor([[ 16.0156, 377.1875],[ 45.9062, 229.5703],[ 59.3750, 459.8750], [ 94.4766, 155.7422], [ 56.5469, 326.3945], [409.6777, 378.8906],
        ...                      [327.7773, 185.5117], [ 27.0625, 156.4453], [121.6250, 452.5000], [402.8477, 412.9609], [196.8281, 156.6094], [202.9375, 293.7734],
        ...                      [139.6562, 324.9727],[ 84.5312, 326.2695], [ 52.7656, 293.5781],[367.6094, 182.1094], [379.1543, 202.5859], [169.1406, 312.4688],
        ...                      [356.0312, 187.2891],[191.6719, 476.9531],[381.6484, 213.8477],[ 91.6328,  92.5391],[ 62.7266, 200.2344],[396.4111, 226.1543]])                
        
        * use the ``'RANSAC'`` to compute E:
        >>> E, mask= find_essential_mat(pts1,pts2,K,method='RANSAC', terminate = 2000, threshold = 0.5)
        >>> E
        tensor([[[  0.0472,  -9.2785,   1.8953],
                 [  9.6489,   0.4045, -17.9023],
                 [ -2.2366,  18.1083,   0.2606]]])
        * use the ``'RANSAC_ADAPT'`` to compute E:
        >>> E, mask= find_essential_mat(pts1,pts2,K,method='RANSAC_ADAPT', terminate = 0.999, threshold = 0.5)
        >>> E
        tensor([[[  0.0472,  -9.2785,   1.8953],
                 [  9.6489,   0.4045, -17.9023],
                 [ -2.2366,  18.1083,   0.2606]]])

        * use all points in the corresponding points set to compute E:

        >>> E = find_essential_mat(pts1,pts2,K)
        >>> E
        tensor([[  0.0472,  -9.2785,   1.8953],
                [  9.6489,   0.4045, -17.9023],
                [ -2.2366,  18.1083,   0.2606]])
        
    """
    assert coordinates1.shape == coordinates2.shape, "Point sets has to be the same shape!"
    assert coordinates1.shape[-1] == 2, "the coordinates shape has to be (..., N, 2)!"
    
    supported_methods = ["none","ransac", "ransac_adapt"]   
    # transfer to float and keep data at same device
    device = coordinates1.device
    
    coordinates1 = coordinates1.type(torch.FloatTensor).to(device)
    coordinates2 = coordinates2.type(torch.FloatTensor).to(device)

    # transfer to homogeneous coordinates
    ones = torch.ones_like(coordinates1[..., :1])
    PH1 = torch.cat([coordinates1, ones], dim = -1)
    PH2 = torch.cat([coordinates2, ones], dim = -1)

    def fit_model(data,samples):
        coordinates1 = data['coordinates_1'][samples[:,:]]
        coordinates2 = data['coordinates_2'][samples[:,:]]
        return eight_pts_alg(coordinates1, coordinates2)
    def check_model(data, M):
        coordinates1 = data['coordinates_1']
        coordinates2 = data['coordinates_2']
        return compute_error(coordinates1, coordinates2, M)

    if method == 'RANSAC':
        ransac = RANSAC('ransac')
        data = {'coordinates_1': PH1, 'coordinates_2': PH2, 'highest_int': len(PH1)}
        num_of_select = 18
        
        F, mask = ransac(data,fit_model,check_model,terminate,num_of_select,threshold)
        E =  intrinsic.T @ F @ intrinsic
        return E, mask
    
    elif method == 'RANSAC_ADAPT':
        ransac = RANSAC('ransac_adapt')
        data = {'coordinates_1': PH1, 'coordinates_2': PH2, 'highest_int': len(PH1)}
        num_of_select = 18
        F, mask = ransac(data,fit_model,check_model,terminate,num_of_select,threshold)
        E =  intrinsic.T @ F @ intrinsic
        return E, mask
    
    elif method == 'none':
        F = eight_pts_alg(PH1,PH2)
        E =  intrinsic.T @ F @ intrinsic
        return E
    else:
        raise NotImplementedError(f"{method} is unknown. Try one of {supported_methods}")

def decompose_essential_mat(E):
    r""" 
    Decompose the essential matrix into possible rotation and translation matrices.
   
    Args:
        E (``torch.Tensor``): The essential matrix with the shape (..., 3, 3).
    
    Returns:
        (``R1: torch.Tensor, R2: torch.Tensor, t: torch.Tensor``):
        ``R1``: The first possible rotation matrix. The shape is (..., 3, 3).

        ``R2``: The second possible rotation matrix. The shape is (..., 3, 3).

        ``t``: The possible translation matrix. The shape is (..., 3, 1).

    Example:
        >>> E = torch.tensor([[  0.0472,  -9.2785,   1.8953],
        ...                   [  9.6489,   0.4045, -17.9023],
        ...                   [ -2.2366,  18.1083,   0.2606]])
        >>> R1,R2,t = decompose_essential_mat(E)
        >>> R1
        tensor([[ 0.9997,  0.0097,  0.0229],
                [-0.0101,  0.9998,  0.0171],
                [-0.0228, -0.0173,  0.9996]])
        >>> R2
        tensor([[ 0.5511,  0.1697,  0.8170],
                [ 0.1858, -0.9795,  0.0781],
                [ 0.8135,  0.1088, -0.5714]])
        >>> t
        tensor([[0.8864],
                [0.1004],
                [0.4519]])

    """
    # keep data at same device
    device = E.device

    W = torch.tensor([0.0, -1.0, 0.0,
                      1.0,  0.0, 0.0, 
                      0.0,  0.0, 1.0]).view(3,3).to(device)
    U,S,VT = torch.linalg.svd(E)

    _, _, t = torch.chunk(U, dim = -1, chunks = 3)
    
    R1 = U @ W @ VT
    R2 = U @ W.T @ VT

    # The rotation matrix has the constraint that the determinant is 1
    if len(R1.shape) == 3:
        R1 = torch.linalg.det(R1).view(-1,1,1) * R1
        R2 = torch.linalg.det(R2).view(-1,1,1) * R2
    
    if len(R1.shape) == 2:
        R1 = torch.linalg.det(R1).view(-1,1) * R1
        R2 = torch.linalg.det(R2).view(-1,1) * R2

    return R1, R2, t

def recover_pose(E,coordinates1:torch.Tensor,coordinates2:torch.Tensor,intrinsic):
    r"""
    Decompose the essential matrix into 4 possible poses,[R1,t],[R1,-t],[R2,t],[R2,-t].
    Return the rotation and translation matrices in which the triangulated points are in front of both cameras.
    
    Args:
        E (``torch.Tensor``): The essential matrix with the shape (..., 3, 3).
        coordinates1 (``torch.Tensor``): Image coordinates with the shape (..., N, 2).
        coordinates2 (``torch.Tensor``): Image coordinates with the shape (..., N, 2).
        intrinsic (``torch.Tensor``): The intrinsic matrix with the shape (..., 3, 3).
    
    Returns:
        (``R: torch.Tensor, t: torch.Tensor, mask: torch.Tensor``):
        ``R``: The rotation matrix with the shape(..., 3, 3). 

        ``t``: The translation matrix with the shape(..., 3, 1). 


    Example:
        >>> K = torch.tensor([[320., 0., 320.], [0., 320., 240.], [0., 0., 1.]])
        >>> pts1 = torch.tensor([[ 40, 368],[ 63, 224], [ 85, 447], [108, 151], [ 77, 319], [411, 371],[335, 183],[ 42, 151],[144, 440],
        ...                      [407, 403],[208, 153], [216, 288], [156, 318], [104, 319], [ 72, 287], [373, 180], [383, 200], [184, 306],
        ...                      [362, 185], [211, 463], [385, 211], [ 80,  81],[ 78, 195], [398, 223]])
        >>> pts2 = torch.tensor([[ 16.0156, 377.1875],[ 45.9062, 229.5703],[ 59.3750, 459.8750], [ 94.4766, 155.7422], [ 56.5469, 326.3945], [409.6777, 378.8906],
        ...                      [327.7773, 185.5117], [ 27.0625, 156.4453], [121.6250, 452.5000], [402.8477, 412.9609], [196.8281, 156.6094], [202.9375, 293.7734],
        ...                      [139.6562, 324.9727],[ 84.5312, 326.2695], [ 52.7656, 293.5781],[367.6094, 182.1094], [379.1543, 202.5859], [169.1406, 312.4688],
        ...                      [356.0312, 187.2891],[191.6719, 476.9531],[381.6484, 213.8477],[ 91.6328,  92.5391],[ 62.7266, 200.2344],[396.4111, 226.1543]])                
        
        >>> E = find_essential_mat(pts1,pts2,K)
        >>> R, t = recover_pose(E,pts1,pts2,K)
        >>> R
        tensor([[ 0.9997,  0.0097,  0.0229],
                [-0.0101,  0.9998,  0.0171],
                [-0.0228, -0.0173,  0.9996]])
        >>> t
        tensor([[-0.8864],
                [-0.1004],
                [-0.4519]])
    """

    R1,R2,t = decompose_essential_mat(E)
    p3d1 = triangulate_points(coordinates1,coordinates2,intrinsic,intrinsic,R= R1, t= t)
    p3d2 = triangulate_points(coordinates1,coordinates2,intrinsic,intrinsic,R= R1, t= -t)
    p3d3 = triangulate_points(coordinates1,coordinates2,intrinsic,intrinsic,R= R2, t= t)
    p3d4 = triangulate_points(coordinates1,coordinates2,intrinsic,intrinsic,R= R2, t= -t)

    # select the combination with the most positive numbers

    _,_,z1 = torch.chunk(p3d1,dim = -1, chunks =3)
    _,_,z2 = torch.chunk(p3d2,dim = -1, chunks =3)
    _,_,z3 = torch.chunk(p3d3,dim = -1, chunks =3)
    _,_,z4 = torch.chunk(p3d4,dim = -1, chunks =3)

    Z = torch.cat((z1,z2,z3,z4),dim = -1).transpose(-2,-1)   # (..., 4, n)
    means = Z.mean(dim = -1)
    max_seq = torch.argmax(means, dim = -1)

    if len(R1.shape) == 3: shape = [-1,1,1]
    if len(R1.shape) == 2: shape = [-1,1]
    
    seq1 = torch.where(max_seq < 2, 1, 0).view(shape)
    seq2 = torch.where(max_seq >= 2, 1, 0).view(shape)
    seq3 = torch.where(((max_seq == 2) + (max_seq ==0)) == True, 1, -1).view(shape)

    R = seq1 * R1 + seq2 * R2
    t = seq3 * t
   
    return R, t

def compute_error(coordinates1:torch.Tensor,coordinates2:torch.Tensor, F):
    r"""
    Finding the epipolar line and caculating the distance (error) between 
    the corroseponding point and epipolarline.
    
    Args:
        coordinates1 (``torch.Tensor``): Image coordinates with the shape (..., N, 2).
        coordinates2 (``torch.Tensor``): Image coordinates with the shape (..., N, 2).
        F (``torch.Tensor``): The fundamental matrix with the shape (..., 3, 3).
        threshold (``float``): The maximum tolerable distance from the corresponding points to the epipolarlines.
        
    Returns:
        (``err: torch.Tensor``, mask: torch.Tensor``):
        ``err``: The distance (error) between the corroseponding points and epipolarlines. The shape is (..., N). 

    Example:
        >>> pts1 = torch.tensor([[ 40, 368],[ 63, 224], [ 85, 447], [108, 151], [ 77, 319], [411, 371],[335, 183],[ 42, 151],[144, 440],
        ...                      [407, 403],[208, 153], [216, 288], [156, 318], [104, 319], [ 72, 287], [373, 180], [383, 200], [184, 306],
        ...                      [362, 185], [211, 463], [385, 211], [ 80,  81],[ 78, 195], [398, 223]]).type(torch.FloatTensor)
        >>> pts2 = torch.tensor([[ 16.0156, 377.1875],[ 45.9062, 229.5703],[ 59.3750, 459.8750], [ 94.4766, 155.7422], [ 56.5469, 326.3945], [409.6777, 378.8906],
        ...                      [327.7773, 185.5117], [ 27.0625, 156.4453], [121.6250, 452.5000], [402.8477, 412.9609], [196.8281, 156.6094], [202.9375, 293.7734],
        ...                      [139.6562, 324.9727],[ 84.5312, 326.2695], [ 52.7656, 293.5781],[367.6094, 182.1094], [379.1543, 202.5859], [169.1406, 312.4688],
        ...                      [356.0312, 187.2891],[191.6719, 476.9531],[381.6484, 213.8477],[ 91.6328,  92.5391],[ 62.7266, 200.2344],[396.4111, 226.1543]])   
        >>> pts1 = pp.cart2homo(pts1)
        >>> pts2 = pp.cart2homo(pts2)
        >>> F = eight_pts_alg(pts1,pts2)
        >>> err = compute_error(pts1,pts2,F)
        >>> err
        tensor([0.0041, 0.0015, 0.0006, 0.0020, 0.0008, 0.0150, 0.0014, 0.0005, 0.0010,
                0.0031, 0.0009, 0.0007, 0.0020, 0.0015, 0.0004, 0.0020, 0.0024, 0.0012,
                0.0015, 0.0026, 0.0017, 0.0018, 0.0003, 0.0038])

    """

    left_epi_lines =  coordinates1 @ F.transpose(-2,-1)  # (..., N, 3)
    right_epi_lines = coordinates2 @ F    # (..., N, 3)

    left_d = torch.sum(left_epi_lines * coordinates2, dim = -1)
    left_a, left_b, _ = torch.chunk(left_epi_lines, dim = -1, chunks = 3)
    left_ab = torch.cat((left_a,left_b),dim =-1)
    left_err = left_d.abs() / torch.linalg.norm(left_ab, dim = -1)

    right_d = torch.sum(right_epi_lines * coordinates1,dim= -1) 
    right_a, right_b, _ = torch.chunk(left_epi_lines, dim = -1, chunks = 3)
    right_ab = torch.cat((right_a,right_b),dim =-1)
    right_err = right_d.abs() / torch.linalg.norm(right_ab, dim = -1)

    err = torch.maximum(left_err,right_err)  # (..., N)
    
    return err