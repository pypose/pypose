import torch
from .. import bmv
from .. import mat2SE3
from ..basics import cart2homo
from ..optim import LM, GN
from .cameras import PerspectiveCameras
from ..optim.scheduler import StopOnPlateau


class BetasOptimizationObjective(torch.nn.Module):
    # Optimize the betas according to the objectives in the paper.
    # For the details, please refer to equation 15.
    def __init__(self, betas):
        super().__init__()
        self.betas = torch.nn.Parameter(betas)

    def forward(self, ctrl_pts_w, kernel_bases):
        # Args:
        #     ctrl_pts_w: The control points in world coordinate. The shape is (B, 4, 3).
        #     kernel_bases: The kernel bases. The shape is (B, 16, 4).
        # Returns:
        #     torch.Tensor: The loss. The shape is (B, ).
        batch_shape = kernel_bases.shape[:-2]
        # calculate the control points in camera coordinate
        ctrl_pts_c = bmv(kernel_bases, self.betas)
        diff_c = ctrl_pts_c.reshape(*batch_shape, 1, 4, 3) - ctrl_pts_c.reshape(*batch_shape, 4, 1, 3)
        diff_c = diff_c.reshape(*batch_shape, 48)  # TODO: whether it is (16, 3) or (48, )?
        diff_c = torch.norm(diff_c, dim=-1)

        # calculate the distance between control points in world coordinate
        diff_w = ctrl_pts_w.reshape(*batch_shape, 1, 4, 3) - ctrl_pts_w.reshape(*batch_shape, 4, 1, 3)
        diff_w = diff_w.reshape(*batch_shape, 48)
        diff_w = torch.norm(diff_w, dim=-1)

        return diff_w - diff_c


class EPnP(torch.nn.Module):
    r"""
    EPnP Solver - a non-iterative O(n) solution to the PnP problem.

    Args:
        optimizer (Optional[torch.optim.Optimizer]): Optimizer to refine the solution.
            Set to ``None`` to disable refinement.
        naive (bool): Use naive control points selection method, otherwise use SVD
            decomposition method to select. Default: ``False``.

    Examples:
        >>> import torch, pypose as pp
        >>> torch.set_default_dtype(torch.float64)
        >>> # create some random test sample for a single camera
        >>> pose = pp.SE3([ 0.0000, -8.0000,  0.0000,  0.0000, -0.3827,  0.0000,  0.9239])
        >>> f, img_size = 2, (9, 9)
        >>> projection = torch.tensor([[f, 0, img_size[0] / 2],
        ...                            [0, f, img_size[1] / 2],
        ...                            [0, 0, 1              ]])
        >>> # some random points in the view
        >>> pts_c = torch.tensor([[2., 0., 2.],
        ...                       [1., 0., 2.],
        ...                       [0., 1., 1.],
        ...                       [0., 0., 1.],
        ...                       [1., 0., 1.],
        ...                       [5., 5., 3.]])
        >>> pixels = (pts_c @ projection.T)[:, :2] / (pts_c @ projection.T)[:, 2:]
        >>> pixels
        tensor([[6.5000, 4.5000],
                [5.5000, 4.5000],
                [4.5000, 6.5000],
                [4.5000, 4.5000],
                [6.5000, 4.5000],
                [7.8333, 7.8333]])
        >>> # transform the points to world coordinate
        >>> # solve the PnP problem to find the camera pose
        >>> pts_w = pose.Inv().Act(pts_c)
        >>> # solve the PnP problem
        >>> epnp = pp.module.EPnP()
        >>> # when input is not batched, remember to add a batch dimension
        >>> solved_pose = epnp(pts_w[None], pixels[None], projection[None])
        >>> assert torch.allclose(solved_pose, pose[None], atol=1e-3)

    Note:
        The implementation is based on the paper

        * Francesc Moreno-Noguer, Vincent Lepetit, and Pascal Fua, `Accurate
          Non-Iterative O(n) Solution to the PnP Problem
          <https://github.com/cvlab-epfl/EPnP>`_, In Proceedings of ICCV, 2007.
    """
    def __init__(self, naive=False, optimizer=GN):
        super().__init__()
        self.naive = naive
        self.optimizer = optimizer

        self.register_buffer('six_indices', torch.tensor(
            [(0 * 4 + 1), (0 * 4 + 2), (0 * 4 + 3), (1 * 4 + 2), (1 * 4 + 3), (2 * 4 + 3)]))
        self.register_buffer('six_indices_pair', torch.tensor([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]).T)
        self.register_buffer('ten_indices_pair', torch.tensor([(0, 0),
                                                               (0, 1), (1, 1),
                                                               (0, 2), (1, 2), (2, 2),
                                                               (0, 3), (1, 3), (2, 3), (3, 3)]).T)
        # equal mask for above pairs [ True, False, True, False, False, True, False, False, False, True, ]
        self.register_buffer('multiply_mask', torch.tensor([1., 2., 1., 2., 2., 1., 2., 2., 2., 1.]))

    def forward(self, points, pixels, intrinsics):
        """
        Args:
            points (``torch.Tensor``): 3D object points in the world coordinates.
                Shape (batch_size, n, 3)
            pixels (``torch.Tensor``): 2D image points, which are the projection of
                object points. Shape (batch_size, n, 2)
            intrinsics (``torch.Tensor``): camera intrinsics. Shape (batch_size, 3, 3)

        Returns:
            ``LieTensor``: estimated pose (``SE3type``) for the camera.
        """
        # shape checking
        batch = torch.broadcast_shapes(points.shape[:-2], pixels.shape[:-2], intrinsics.shape[:-2])

        # Select naive and calculate alpha in the world coordinate
        bases = self.naive_basis(points) if self.naive else self.svd_basis(points)
        alpha = self.compute_alphas(points, bases)
        m = self.build_m(pixels, alpha, intrinsics)

        kernel_m = self.calculate_kernel(m)[..., [3, 2, 1, 0]]  # to be consistent with the matlab code

        l_mat = self.build_l(kernel_m)
        rho = self.build_rho(bases)

        solution_keys = ['error', 'pose', 'ctrl_pts_c', 'points_c', 'beta', 'scale']
        solutions = {key: [] for key in solution_keys}
        for dim in range(1, 4):  # kernel space dimension of 4 is unstable
            beta = self.calculate_betas(dim, l_mat, rho)
            solution = self.generate_solution(beta, kernel_m, alpha, points, pixels, intrinsics)
            for key in solution_keys:
                solutions[key].append(solution[key])

        # stack the results
        for key in solutions.keys():
            solutions[key] = torch.stack(solutions[key], dim=len(batch))
        best_error, best_idx = torch.min(solutions['error'], dim=len(batch))
        for key in solutions.keys():
            # retrieve the best solution using gather
            best_idx_ = best_idx.reshape(best_idx.shape + (1,) * (solutions[key].dim() - len(batch)))
            best_idx_ = best_idx_.tile((1,) * (len(batch) + 1) + solutions[key].shape[len(batch) + 1:])
            solutions[key] = torch.gather(solutions[key], len(batch), best_idx_)
            solutions[key] = solutions[key].squeeze(len(batch))

        if self.optimizer is not None:
            solutions = self.optimization_step(solutions, kernel_m, bases, alpha, points, pixels, intrinsics)
        return solutions['pose']

    def generate_solution(self, beta, kernel_m, alpha, points, pixels, intrinsics, request_error=True):
        solution = dict()

        ctrl_pts_c = bmv(kernel_m, beta)
        ctrl_pts_c, points_c, sc = self.compute_norm_sign_scaling_factor(ctrl_pts_c, alpha, points)
        pose = self.get_se3(points, points_c)
        if request_error:
            perspective_camera = PerspectiveCameras(pose, intrinsics)
            error = perspective_camera.reprojection_error(points, pixels)
            solution['error'] = error

        # save the solution
        solution['pose'] = pose
        solution['ctrl_pts_c'] = ctrl_pts_c
        solution['points_c'] = points_c
        solution['beta'] = beta
        solution['scale'] = sc

        return solution

    def optimization_step(self, solutions, kernel_m, ctrl_pts_w, alpha, points, pixels, intrinsics):
        """
        Args:
            solutions (dict): a dict of solutions
            kernel_m (Tensor): kernel matrix, shape (batch_size, n, 4)
            ctrl_pts_w (Tensor): control points in the world coordinate, shape (batch_size, 4, 3)
            alpha (Tensor): alpha, shape (batch_size, n, 4)
            points (Tensor): 3D object points, shape (batch_size, n, 3)
            pixels (Tensor): 2D image points, shape (batch_size, n, 2)
            intrinsics (Tensor): camera intrinsics, shape (batch_size, 3, 3)
        Returns:
            None. This function will update the solutions in place.
        """
        objective = BetasOptimizationObjective(solutions['beta'] * solutions['scale'].unsqueeze(-1))
        gn = self.optimizer(objective)
        scheduler = StopOnPlateau(gn, steps=10, patience=3, verbose=False)
        scheduler.optimize(input=(ctrl_pts_w, kernel_m))
        beta = objective.betas.data

        solution = self.generate_solution(beta, kernel_m, alpha, points, pixels, intrinsics, request_error=False)
        return solution

    @staticmethod
    def naive_basis(points):
        # Select 4 naive points, 3 unit bases and 1 origin point.
        controls = torch.zeros_like(points[...,:4,:])
        controls.diagonal(dim1=-2, dim2=-1).fill_(1)
        return controls

    @staticmethod
    def svd_basis(points):
        # Select 4 control points with SVD
        center = points.mean(dim=-2, keepdim=True)
        translated = points - center
        u, s, vh = torch.linalg.svd(translated.mT @ translated)
        controls = center + s.sqrt().unsqueeze(-1) * vh.mT
        return torch.cat([center, controls], dim=-2)


    @staticmethod
    def compute_alphas(points, bases):
        # Compute the coordinates of points with respect to the bases.
        # Check equation 1 in paper for more details.
        points, bases = cart2homo(points), cart2homo(bases)
        return torch.linalg.solve(A=bases, B=points, left=False)

    @staticmethod
    def build_m(pixels, alpha, intrinsics):
        # Construct M matrix. Check equation 7 in paper for more details.
        # Args:
        #     pixels (torch.Tensor): (..., point, 2)
        #     alpha (torch.Tensor): (..., point, 4)
        #     intrinsics (torch.Tensor): (..., 3, 3)
        # Returns:
        #     torch.Tensor: (..., point * 2, 12)
        batch, point = pixels.shape[:-2], pixels.shape[-2]
        u, v = pixels[..., 0], pixels[..., 1]
        fu, u0 = intrinsics[..., 0, 0, None], intrinsics[..., 0, 2, None]
        fv, v0 = intrinsics[..., 1, 1, None], intrinsics[..., 1, 2, None]
        a0, a1, a2, a3 = alpha[..., 0], alpha[..., 1], alpha[..., 2], alpha[..., 3]
        O = torch.zeros_like(a1)
        M = torch.stack([a0 * fu, O, a0 * (u0 - u),
                         a1 * fu, O, a1 * (u0 - u),
                         a2 * fu, O, a2 * (u0 - u),
                         a3 * fu, O, a3 * (u0 - u),
                         O, a0 * fv, a0 * (v0 - v),
                         O, a1 * fv, a1 * (v0 - v),
                         O, a2 * fv, a2 * (v0 - v),
                         O, a3 * fv, a3 * (v0 - v)], dim=-1) # (batch, point, 24)
        return M.reshape(*batch, point * 2, 12)

    @staticmethod
    def calculate_kernel(M, least=4):
        # Given M matrix, find eigenvectors with the least eigenvalues.
        # Check equation 8 in paper for more details.
        batch = M.shape[:-2] # M is (..., point * 2, 12)
        eigenvalues, eigenvectors = torch.linalg.eig(M.mT @ M)
        eigenvalues, eigenvectors = eigenvalues.real, eigenvectors.real
        index = eigenvalues.argsort(descending=False)[..., :least] # (batch, 4)
        index = index.unsqueeze(-2).tile((1,) * len(batch) + (12, 1)) # (batch, 12, 4)
        return torch.gather(eigenvectors, dim=-1, index=index) # (batch, 12, 4)

    def build_l(self, kernel_bases):
        """Given the kernel of m, compute the L matrix. Check [source]
        (https://github.com/cvlab-epfl/EPnP/blob/5abc3cfa76e8e92e5a8f4be0370bbe7da246065e/cpp/epnp.cpp#L478)
        for more details. Inputs are batched.

        Args:
            kernel_bases (torch.Tensor): kernel of m, shape (..., 12, 4)
        Returns:
            torch.Tensor: L, shape (..., 6, 10)
        """
        batch_shape = kernel_bases.shape[:-2]
        kernel_bases = kernel_bases.mT  # shape (batch_shape, 4, 12)
        # calculate the pairwise distance matrix within bases
        diff = kernel_bases.reshape(*batch_shape, 4, 1, 4, 3) - kernel_bases.reshape(*batch_shape, 4, 4, 1, 3)
        diff = diff.flatten(start_dim=-3, end_dim=-2)  # shape (batch_shape, 4, 16, 3)
        # six_indices are (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3) before flatten
        dv = diff[..., self.six_indices, :]  # shape (batch_shape, 4, 6, 3)

        # generate l
        dot_products = torch.sum(
            dv[..., self.ten_indices_pair[0], :, :] * dv[..., self.ten_indices_pair[1], :, :], dim=-1)
        dot_products = dot_products * self.multiply_mask.reshape((1,) * len(batch_shape) + (10, 1))
        return dot_products.mT  # shape (batch_shape, 6, 10)

    def build_rho(self, cont_pts_w):
        """Given the coordinates of control points, compute the rho vector. Check [source]
        (https://github.com/cvlab-epfl/EPnP/blob/5abc3cfa76e8e92e5a8f4be0370bbe7da246065e/cpp/epnp.cpp#L520)
        for more details.
        Inputs are batched.

        Args:
            cont_pts_w (torch.Tensor): coordinates of control points, shape (..., 4, 3)
        Returns:
            torch.Tensor: rho, shape (..., 6, 1)
        """
        dist = cont_pts_w[..., self.six_indices_pair[0], :] - cont_pts_w[..., self.six_indices_pair[1], :]
        return torch.sum(dist ** 2, dim=-1)  # l2 norm

    @staticmethod
    def calculate_betas(dim, l_mat, rho):
        """Given the L matrix and rho vector, compute the beta vector. Check equation 10 - 14 in paper for more details.
        Inputs are batched.

        Args:
            dim (int): dimension of the problem, 1, 2, or 3
            l_mat (torch.Tensor): L, shape (..., 6, 10)
            rho (torch.Tensor): rho, shape (..., 6)
        Returns:
            torch.Tensor: beta, shape (..., 4)
        """
        batch_shape = l_mat.shape[:-2]
        if dim == 1:
            betas = torch.zeros(*batch_shape, 4, device=l_mat.device, dtype=l_mat.dtype)
            betas[..., -1] = 1
            return betas
        elif dim == 2:
            l_mat = l_mat[..., (5, 8, 9)]  # matched with matlab code
            betas_ = bmv(torch.linalg.pinv(l_mat), rho)  # shape: (b, 3)
            beta1 = torch.sqrt(torch.abs(betas_[..., 0]))
            beta2 = torch.sqrt(torch.abs(betas_[..., 2])) * torch.sign(betas_[..., 1]) * torch.sign(betas_[..., 0])

            return torch.stack([torch.zeros_like(beta1), torch.zeros_like(beta1), beta1, beta2], dim=-1)
        elif dim == 3:
            l_mat = l_mat[..., (2, 4, 7, 5, 8, 9)]  # matched with matlab code
            betas_ = torch.linalg.solve(l_mat, rho.unsqueeze(-1)).squeeze(-1)  # shape: (b, 6)
            beta1 = torch.sqrt(torch.abs(betas_[..., 0]))
            beta2 = torch.sqrt(torch.abs(betas_[..., 3])) * torch.sign(betas_[..., 1]) * torch.sign(betas_[..., 0])
            beta3 = torch.sqrt(torch.abs(betas_[..., 5])) * torch.sign(betas_[..., 2]) * torch.sign(betas_[..., 0])

            return torch.stack([torch.zeros_like(beta1), beta1, beta2, beta3], dim=-1)
        elif dim == 4:
            betas_ = bmv(torch.linalg.pinv(l_mat), rho)  # shape: (b, 10)
            beta4 = torch.sqrt(abs(betas_[..., 0]))
            beta3 = torch.sqrt(abs(betas_[..., 2])) * torch.sign(betas_[..., 1]) * torch.sign(betas_[..., 0])
            beta2 = torch.sqrt(abs(betas_[..., 5])) * torch.sign(betas_[..., 3]) * torch.sign(betas_[..., 0])
            beta1 = torch.sqrt(abs(betas_[..., 9])) * torch.sign(betas_[..., 6]) * torch.sign(betas_[..., 0])

            return torch.stack([beta1, beta2, beta3, beta4], dim=-1)

    @staticmethod
    def compute_norm_sign_scaling_factor(xc, alphas, points):
        """Compute the scaling factor and the sign of the scaling factor

        Args:
            xc (torch.tensor): the (unscaled) control points in the camera coordinates, or the result from null space.
            alphas (torch.tensor): the weights of the control points to recover the object points
            points (torch.tensor): the object points in the world coordinates
        Returns:
            contPts_c (torch.tensor): the control points in the camera coordinates
            objPts_c (torch.tensor): the object points in the camera coordinates
            sc (torch.tensor): the scaling factor
        """
        batch_shape = xc.shape[:-1]
        # Calculate the control points and object points in the camera coordinates
        ctrl_pts_c = xc.reshape((*batch_shape, 4, 3))
        points_c = alphas @ ctrl_pts_c

        # Calculate the distance of the reference points in the world coordinates
        points_w_centered = points - points.mean(dim=-2, keepdim=True)
        dist_w = torch.linalg.norm(points_w_centered, dim=-1)

        # Calculate the distance of the reference points in the camera coordinates
        points_c_centered = points_c - points_c.mean(dim=-2, keepdim=True)
        dist_c = torch.linalg.norm(points_c_centered, dim=-1)

        # calculate the scaling factors
        # below are batched vector dot product
        sc_1 = torch.matmul(dist_c.unsqueeze(-2), dist_c.unsqueeze(-1))
        sc_2 = torch.matmul(dist_c.unsqueeze(-2), dist_w.unsqueeze(-1))
        sc = (1 / sc_1 * sc_2)

        # Update the control points and the object points in the camera coordinates based on the scaling factors
        ctrl_pts_c = ctrl_pts_c * sc
        points_c = alphas.matmul(ctrl_pts_c)

        # Update the control points and the object points in the camera coordinates based on the sign
        neg_z_mask = torch.any(points_c[..., 2] < 0, dim=-1)  # (N, )
        negate_switch = torch.ones((points.shape[0],), dtype=points.dtype, device=points.device)
        negate_switch[neg_z_mask] = negate_switch[neg_z_mask] * -1
        points_c = points_c * negate_switch.reshape(*batch_shape, 1, 1)
        sc = sc[..., 0, 0] * negate_switch
        return ctrl_pts_c, points_c, sc

    @staticmethod
    def get_se3(pts_w, pts_c):
        """
        Get the rotation matrix and translation vector based on the object points in world coordinate and camera
        coordinate.

        Args:
            pts_w: The object points in world coordinate. The shape is (..., N, 3).
            pts_c: The object points in camera coordinate. The shape is (..., N, 3).
        Returns:
            R: The rotation matrix. The shape is (..., 3, 3).
            T: The translation vector. The shape is (..., 3).
        """
        # Get the centered points
        center_w = pts_w.mean(dim=-2)
        pts_w = pts_w - center_w.unsqueeze(-2)
        center_c = pts_c.mean(dim=-2)
        pts_c = pts_c - center_c.unsqueeze(-2)

        # Calculate the rotation matrix
        m = torch.matmul(pts_c[..., :, None], pts_w[..., None, :])
        m = m.sum(dim=1)  # along the point dimension

        u, s, vh = torch.svd(m)
        rot = u.matmul(vh.mT)

        # if det(R) < 0, make it positive
        negate_mask = torch.linalg.det(rot) < 0
        rot[negate_mask] = -rot[negate_mask]

        # Calculate the translation vector based on the rotation matrix and the equation
        t = center_c - bmv(rot, center_w)
        rt = torch.cat((rot, t.unsqueeze(-1)), dim=-1)
        pose = mat2SE3(rt)

        return pose

