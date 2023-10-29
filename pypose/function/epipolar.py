import torch
import pypose as pp

def normalizePoints(coordinates:torch.Tensor):
    r"""  
    Calculate the normalized transformation of each coordinate set. After the transformation, 
    the centroid of each coordinate set is located at the coordinate origin, 
    and the average distance from the origin is sqrt(2).
    Args:
        coordinates (``torch.Tensor``): Homogeneous coordinates with the shape (N,3).

    Returns:
        normP (``torch.Tensor``): normalized coordinates with the shape (N,3).
        transform (``torch.Tensor``): transformation matrix with the shape (3,3).
    
    Example:


    """
    assert coordinates.shape[1] == 3, "the coordinates has to be the homogeneous!"
    
    # keep data at same device
    device = coordinates.device
    if coordinates.type() != 'torch.FloatTensor':
        coordinates = coordinates.type(torch.FloatTensor).to(device)  

    mean = torch.mean(coordinates, dim = 0)
    scale = 1.4142135623730950488 / (torch.linalg.norm(coordinates - mean, dim = 1).mean() + 1e-8)

    transform = torch.tensor([scale,     0.0, - scale * mean [0],
                                0.0,   scale, - scale * mean [1],
                                0.0,     0.0, 1.0 ]).view(3,3).to(device)  

    normP = transform @ coordinates.T
    return normP.T, transform

def triangulatePoints(coordinates1,coordinates2,intrinsic1,intrinsic2,R,t):
    r"""
    Convert 2D corresponding coordinates to 3D points.
    
    Args:
        coordinates1 (``torch.Tensor``): Image coordinates with the shape (N, 2).
        coordinates2 (``torch.Tensor``): Image coordinates with the shape (N, 2).
        intrinsic1 (``torch.Tensor``): The intrinsic matrix of the first camera with the shape (3, 3).
        intrinsic2 (``torch.Tensor``): The intrinsic matrix of the second camera with the shape (3, 3).
        R (``torch.Tensor``): The rotation matrix with the shape(3, 3). 
        t (``torch.Tensor``): The translation matrix with the shape(3, 1). 
    
    Returns:
        points3D (``torch.Tensor``): The 3D points with the shape(N, 3). 

    Example:

    """
    M = torch.cat([intrinsic1,torch.zeros_like(intrinsic1[..., :1])],dim =-1)
    Proj = intrinsic2 @ torch.cat([R,t],dim =-1)

    # consturct the A and b matrices
    row1 = coordinates1[:,0].view(-1,1) * M[2] - M[0]           # (N, 4)
    row2 = coordinates1[:,1].view(-1,1) * M[2] - M[1]           # (N, 4)
    row3 = coordinates2[:,0].view(-1,1) * Proj[2] - Proj[0]     # (N, 4)
    row4 = coordinates2[:,1].view(-1,1) * Proj[2] - Proj[1]     # (N, 4)
    
    A = torch.cat([row1[:,:3],row2[:,:3],row3[:,:3],row4[:,:3]],dim =-1).view(-1,4,3)     # (N, 4, 3)
    AT = torch.cat([row1[:,:3].reshape(-1,1),row2[:,:3].reshape(-1,1),row3[:,:3].reshape(-1,1),row4[:,:3].reshape(-1,1)],dim =-1).view(-1,3,4)     # (N, 3, 4)
    b = - torch.cat([row1[:,3].reshape(-1,1),row2[:,3].reshape(-1,1),row3[:,3].reshape(-1,1),row4[:,3].reshape(-1,1)],dim =-1).view(-1,4,1)        # (N, 4, 1)
    
    points3D = torch.linalg.inv(AT @ A) @ AT @ b     # (N, 3, 1)

    return points3D.squeeze()

def eightPointsAlg(coordinates1,coordinates2):
    r"""
    A minimum of eight corresponding points from two images 
    to obtain an initial estimate of the essential or fundamental matrix. 

    Args:
        coordinates1 (``torch.Tensor``): Homogeneous coordinates with the shape (N, 3).
        coordinates2 (``torch.Tensor``): Homogeneous coordinates with the shape (N, 3).
    
    Returns:
        F (``torch.Tensor``): essential or fundamental matrix with the shape (3, 3).
    
    Example:

    """
    assert coordinates1.shape == coordinates2.shape, "Point sets has to be the same shape!"
    assert len(coordinates1) > 7, "The number of point pairs must be greater than 7!"

    PH1, Norm_matrix1 = normalizePoints(coordinates1)
    PH2, Norm_matrix2 = normalizePoints(coordinates2)

    x1,y1 = PH1[:,0].view(-1,1),PH1[:,1].view(-1,1)
    x2,y2 = PH2[:,0].view(-1,1),PH2[:,1].view(-1,1)

    A = torch.cat([x2*x1,x2*y1,x2,y2*x1,y2*y1,y2,x1,y1,torch.ones_like(PH1[..., :1])], dim=-1)
    _,_,VT = torch.linalg.svd(A.T @ A)
    M = VT[-1,:].view(3,3)

    # Enforce rank 2 of F, put the last singular value to 0
    U,S,VT = torch.linalg.svd(M)
    S[-1] = 0.0
    M_new = U @ torch.diag_embed(S) @ VT

    F = Norm_matrix2.T @ M_new @ Norm_matrix1

    # Make the value of the last row and column is one.
    if abs(F[2,2]) > 0.0:
        F = F/F[-1,-1]
    return F

def ransac(coordinates1,coordinates2,iterations = 1000, threshold = 1):
    r""" 
    The algorithm identifies inliers and outliers. It also caculates the essential or fundamental matrix with the inliers.
   
    Args:
        coordinates1 (``torch.Tensor``): Homogeneous coordinates with the shape (N, 3).
        coordinates2 (``torch.Tensor``): Homogeneous coordinates with the shape (N, 3).
        iterations (``int``, optional): The maximum number of iterations.
        threshold (``float``, optional): The maximum tolerable distance from the corresponding points to the epipolarlines.
    
    Returns:
        Fbest (``torch.Tensor``): Essential or fundamental matrix with the most inliers. The shape is (3, 3).
        Maskbest (``torch.Tensor``): The index of the inliers in the corresponding points set. The shape is (..., 1).

    Example:

    """
    MaxNumOfIns = 8
    
    for i in range(iterations):
        # randomly choose 8-12 samples
        num = torch.randint(0,5,(1,))
        sample = torch.randint(0,len(coordinates1),(8 + num,)) 
        # run 8 points algorithm
        F = eightPointsAlg(coordinates1[sample],coordinates2[sample])
        
        #caculating the distance (error) between the corroseponding point and epipolarline
        err = computeError(coordinates1,coordinates2,F)
        mask = torch.argwhere(err < threshold)

        # err = np.sum( coordinates2 @ F * coordinates1,axis= 1)
        # mask = np.argwhere(abs(err)<threshold) 
        if len(mask) > MaxNumOfIns:
            MaxNumOfIns = len(mask)
            Maskbest = mask
            # refine the F with all inliers
            F = eightPointsAlg(coordinates1[mask[:,0]],coordinates2[mask[:,0]])
            Fbest = F
    return Fbest, Maskbest

def findEssentialMat(coordinates1,coordinates2,intrinsic,method = 'none',iterations =10000, threshold = 0.5):
    r""" 
    Comupte the essential matrix.

    Args:
        coordinates1 (``torch.Tensor``): Image coordinates with the shape (N, 2).
        coordinates2 (``torch.Tensor``): Image coordinates with the shape (N, 2).
        intrinsic (``torch.Tensor``): The intrinsic matrix with the shape (3, 3).
        method (``str``, optional): The method to calculate the fundamental matrix.
                              ``'none'``: No method is applied. This method will compute the E or F with all points in the corresponding points set.
                              ``'RANSAC'``: RANSAC method is applied.

        iterations (``int``, optional): The maximum number of iterations in RANSAC method.
        threshold (``float``, optional): The maximum tolerable distance from the corresponding points to the epipolarlines.
                                         It is only used in RANSAC method. 
    
    Returns:
        E (``torch.Tensor``): Essential matrix with the shape (3, 3).
        mask (``torch.Tensor``): For 'RANSAC', it will return the index of the inliers in the corresponding points set.
                                 The shape is (..., 1)
                                 For 'none', no mask will be returned.
    
    Example:
        use the 'RANSAC' to compute E:

        >> E, mask= findEssentialMat(coordinates1,coordinates2,intrinsic,method='RANSAC', iterations = 1000, threshold = 0.5)

        use all points in the corresponding points set to compute E:

        >> E = findEssentialMat(coordinates1,coordinates2,intrinsic)
        
    """
    assert coordinates1.shape == coordinates2.shape, "Point sets has to be the same shape!"
    assert coordinates1.shape[1] == 2, "the coordinates shape has to be (N, 2)!"
    
    # transfer to float and keep data at same device
    device = coordinates1.device
    if coordinates1.type() != 'torch.FloatTensor':
        coordinates1 = coordinates1.type(torch.FloatTensor).to(device)
    if coordinates2.type() != 'torch.FloatTensor':
        coordinates2 = coordinates2.type(torch.FloatTensor).to(device)

    # transfer to homogeneous coordinates
    PH1 = pp.cart2homo(coordinates1)
    PH2 = pp.cart2homo(coordinates2)

    if method == 'RANSAC':
        F, mask = ransac(PH1,PH2,iterations,threshold)
        E =  intrinsic.T @ F @ intrinsic
        return E, mask
    else:
        F = eightPointsAlg(PH1,PH2)
        E =  intrinsic.T @ F @ intrinsic
        return E

def decomposeEssentialMat(E):
    r""" 
    Decompose the essential matrix into possible rotation and translation matrices.
   
    Args:
        E (``torch.Tensor``): The essential matrix with the shape (3, 3).
    
    Returns:
        R1 (``torch.Tensor``): The first possible rotation matrix. The shape is (3, 3).
        R2 (``torch.Tensor``): The second possible rotation matrix. The shape is (3, 3).
        t (``torch.Tensor``): The possible translation matrix. The shape is (3, 1).

    Example:

    """
    # keep data at same device
    device = E.device

    W = torch.tensor([0.0, -1.0, 0.0,
                      1.0,  0.0, 0.0, 
                      0.0,  0.0, 1.0]).view(3,3).to(device)
    U,S,VT = torch.linalg.svd(E)
    t = U[:,-1].view(-1,1)
    R1 = U @ W @ VT

    # The rotation matrix has the constraint that the determinant is 1
    if torch.linalg.det(R1)<0:
        R1 = -R1

    R2 = U @ W.T @ VT
    if torch.linalg.det(R2)<0:
        R2 = -R2

    return R1,R2,t

def recoverPose(E,coordinates1,coordinates2,intrinsic):
    r"""
    Decompose the essential matrix into 4 possible poses,[R1,t],[R1,-t],[R2,t],[R2,-t].
    Return the rotation and translation matrices in which the triangulated points are in front of both cameras.
    
    Args:
        E (``torch.Tensor``): The essential matrix with the shape (3, 3).
        coordinates1 (``torch.Tensor``): Image coordinates with the shape (N, 2).
        coordinates2 (``torch.Tensor``): Image coordinates with the shape (N, 2).
        intrinsic (``torch.Tensor``): The intrinsic matrix with the shape (3, 3).
    
    Returns:
        R (``torch.Tensor``): The rotation matrix with the shape(3, 3). 
        t (``torch.Tensor``): The translation matrix with the shape(3, 1). 
        mask (``torch.Tensor``): The index of the points in front of both cameras. The shape is (..., 1).

    Example:

    """

    R1,R2,t = decomposeEssentialMat(E)
    p3d1 = triangulatePoints(coordinates1,coordinates2,intrinsic,intrinsic,R= R1, t= t)
    p3d2 = triangulatePoints(coordinates1,coordinates2,intrinsic,intrinsic,R= R1, t= -t)
    p3d3 = triangulatePoints(coordinates1,coordinates2,intrinsic,intrinsic,R= R2, t= t)
    p3d4 = triangulatePoints(coordinates1,coordinates2,intrinsic,intrinsic,R= R2, t= -t)

    # select the combination with the most positive numbers
    mask1 = torch.argwhere(p3d1[:,-1] > 0)
    mask2 = torch.argwhere(p3d2[:,-1] > 0)
    mask3 = torch.argwhere(p3d3[:,-1] > 0)
    mask4 = torch.argwhere(p3d4[:,-1] > 0)
    maxnum = max(len(mask1),len(mask2),len(mask3),len(mask4)) 
    if len(mask1) == maxnum:
        return R1, t, mask1
    if len(mask2) == maxnum:
        return R1,-t, mask2
    if len(mask3) == maxnum:
        return R2, t, mask3 
    return R2, -t, mask4
    
def computeError(coordinates1, coordinates2, F):
    r"""
    Finding the epipolar line and caculating the distance (error) between 
    the corroseponding point and epipolarline.
    
    Args:
        coordinates1 (``torch.Tensor``): Homogeneous coordinates with the shape (N, 3).
        coordinates2 (``torch.Tensor``): Homogeneous coordinates with the shape (N, 3).
        F (``torch.Tensor``): The fundamental matrix with the shape (3, 3).
    
    Returns:
        err (``torch.Tensor``): The distance (error) between the corroseponding points and epipolarlines. 
                                  The shape is (N, 1).

    Example:

    """

    left_epi_lines =  coordinates1 @ F.T  # N*3
    right_epi_lines = coordinates2 @ F    # N*3

    left_d = torch.sum(left_epi_lines * coordinates2,dim= 1)
    left_err = left_d**2 / (left_epi_lines[:,0]**2 +left_epi_lines[:,1]**2)

    right_d = torch.sum(right_epi_lines * coordinates1,dim= 1) # N*1
    right_err = right_d**2 / (right_epi_lines[:,0]**2 + right_epi_lines[:,1]**2)

    err = torch.maximum(left_err,right_err)

    return torch.sqrt(err)