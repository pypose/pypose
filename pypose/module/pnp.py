import torch
from .. import bmv
from .. import mat2SE3
from ..basics import cart2homo
from ..optim import LM, GN
from torch import broadcast_shapes
from .cameras import PerspectiveCameras
from ..optim.scheduler import StopOnPlateau

class BetasOptimizationObjective(torch.nn.Module):
    # Optimize the betas according to the objectives in the ePnP paper.
    # For the details, please refer to equation 15.
    def __init__(self, betas):
        super().__init__()
        self.betas = torch.nn.Parameter(betas)

    def forward(self, ctrl_pts_w, kernel_bases):
        """
        Args:
            ctrl_pts_w: The control points in world coordinate. The shape is (B, 4, 3).
            kernel_bases: The kernel bases. The shape is (B, 16, 4).
        
        Returns:
            torch.Tensor: The loss. The shape is (B, ).
        """
        batch = kernel_bases.shape[:-2]
        # calculate the control points in camera coordinate
        ctrl_pts_c = bmv(kernel_bases, self.betas)
        diff_c = ctrl_pts_c.reshape(*batch, 1, 4, 3) - ctrl_pts_c.reshape(*batch, 4, 1, 3)
        diff_c = diff_c.reshape(*batch, 48)  # TODO: whether it is (16, 3) or (48, )?
        diff_c = torch.norm(diff_c, dim=-1)

        # calculate the distance between control points in world coordinate
        diff_w = ctrl_pts_w.reshape(*batch, 1, 4, 3) - ctrl_pts_w.reshape(*batch, 4, 1, 3)
        diff_w = diff_w.reshape(*batch, 48)
        diff_w = torch.norm(diff_w, dim=-1)

        return diff_w - diff_c


class EPnP(torch.nn.Module):
    r"""
    EPnP Solver - a non-iterative O(n) solution to the PnP problem for :math:`n \geq 4`.

    As an overview of the process, first define each of the n points in the world coordinate as :math:`p^w_i` and their
    corresponding position in camera coordinate as :math:`p^c_i`.
    They are represented by weighted sums of the four selected controls points, :math:`c^w_j` and :math:`c^c_j` in
    camera coordinate respectively. Let the projection matrix be :math:`P = K[R|T]`, where :math:`K` is the camera
    intrinsic matrix, :math:`R` is the rotation matrix and :math:`T` is the translation vector.

    .. math::
        \begin{aligned}
            p^w_i &= \sum^4_{j=1}{\alpha_{ij}c^w_j} \\
            p^c_i &= \sum^4_{j=1}{\alpha_{ij}c^c_j} \\
            \sum^4_{j=1}{\alpha_{ij}} &= 1
        \end{aligned}

    From this, the derivation of the points projection equations is as follows:

    .. math::
        \begin{aligned}
            s_i\,p^{img}_i &= K\sum^4_{j=1}{\alpha_{ij}c^c_j}
        \end{aligned}

    Where :math:`p^{img}_i` is the projected image pixels in form :math:`(u_i, v_i, 1)`, :math:`s_i` is the scale factor.
    Let the control point in camera coordinate represented by :math:`c^c_j = (x^c_j, y^c_j, z^c_j)`.
    Rearranging the image point equation yields the following two linear equations for each of the n points:

    .. math::
        \begin{aligned}
            \sum^4_{j=1}{\alpha_{ij}f_xx^c_j + \alpha_{ij}(u_0 - u_i)z^c_j} &= 0 \\
            \sum^4_{j=1}{\alpha_{ij}f_yy^c_j + \alpha_{ij}(v_0 - v_i)z^c_j} &= 0
        \end{aligned}

    Using these two equations for each of the n points, the system :math:`Mx = 0` can be formed where
    :math:`x = \begin{bmatrix}c^{c^T}_1 & c^{c^T}_2 & c^{c^T}_3 & c^{c^T}_4\end{bmatrix}^T`.
    The solution for the control points exists in the kernel space of :math:`M` and is expressed as

    .. math::
        \begin{aligned}
            x &= \sum^N_{i=1}{\beta_iv_i}
        \end{aligned}

    where N (4) is the number of singular values in M and each :math:`v_i` is the corresponding right singular vector of
    :math:`M`. The final step involves calculating the coefficients :math:`\beta_i`.
    Optionally, the Gauss-Newton algorithm is used to refine them.
    The camera pose that minimize the error of transforming the world coordinat points, :math:`p^w_i`, to image
    coordinate points, :math:`p^c_i` which is known as long as :math:`c^{c^T}_4` is known, are then calculated.

    Args:
        optimizer (Optional[torch.optim.Optimizer]): Optimizer to refine the solution.
            Set to ``None`` to disable refinement.
        naive (bool): Use naive control points selection method, otherwise use SVD
            decomposition method to select. Default: ``False``.
        intrinsics (Optional[torch.Tensor]): The camera intrinsics. The shape is (3, 3).

    Examples:
        >>> import torch, pypose as pp
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
        >>> pixels = pp.homo2cart(pts_c @ projection.T)
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
        >>> epnp = pp.module.EPnP(intrinsics=projection)
        >>> pose = epnp(pts_w, pixels)
        >>> pose
        SE3Type LieTensor:
        LieTensor([ 7.3552e-05, -8.0000e+00,  1.4997e-04, -8.1382e-06, -3.8271e-01,
                    5.6476e-06,  9.2387e-01])

    Note:
        The implementation is based on the paper

        * Francesc Moreno-Noguer, Vincent Lepetit, and Pascal Fua, `Accurate
          Non-Iterative O(n) Solution to the PnP Problem
          <https://github.com/cvlab-epfl/EPnP>`_, In Proceedings of ICCV, 2007.
    """

    def __init__(self, naive=False, optimizer=GN, intrinsics=None):
        super().__init__()
        self.naive = naive
        self.optimizer = optimizer
        if intrinsics is not None:
            self.register_buffer('intrinsics', intrinsics)

        self.register_buffer('six_indices', torch.tensor(
            [(0 * 4 + 1), (0 * 4 + 2), (0 * 4 + 3), (1 * 4 + 2), (1 * 4 + 3), (2 * 4 + 3)]))
        self.register_buffer('six_indices_pair', torch.tensor([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]).T)
        self.register_buffer('ten_indices_pair', torch.tensor([(0, 0),
                                                               (0, 1), (1, 1),
                                                               (0, 2), (1, 2), (2, 2),
                                                               (0, 3), (1, 3), (2, 3), (3, 3)]).T)
        # equal mask for above pairs [ True, False, True, False, False, True, False, False, False, True, ]
        self.register_buffer('multiply_mask', torch.tensor([1., 2., 1., 2., 2., 1., 2., 2., 2., 1.]))


    def forward(self, points, pixels, intrinsics=None):
        r"""
        Args:
            points (``torch.Tensor``): 3D object points in the world coordinates.
                Shape (..., n, 3)
            pixels (``torch.Tensor``): 2D image points, which are the projection of
                object points. Shape (..., n, 2)
            intrinsics (``Optional[torch.Tensor]``): camera intrinsics. Shape (..., 3, 3).
                Setting it to any non-``None`` value will override the default intrinsics
                kept in the module.

        Returns:
            ``LieTensor``: estimated pose (``SE3type``) for the camera.
        """
        intrinsics = self.intrinsics if intrinsics is None else intrinsics
        batch = broadcast_shapes(points.shape[:-2], pixels.shape[:-2], intrinsics.shape[:-2])

        # Select naive and calculate alpha in the world coordinate
        bases = self._naive_basis(points) if self.naive else self._svd_basis(points)
        alpha = self._compute_alphas(points, bases)
        kernel = self._calculate_kernel(pixels, alpha, intrinsics)
        l_mat = self._build_l(kernel)
        rho = self._build_rho(bases)

        solution_keys = ['error', 'pose', 'ctrl_pts_c', 'points_c', 'beta', 'scale']
        solutions = {key: [] for key in solution_keys}
        for dim in range(1, 5):  # kernel space dimension of 4 is unstable
            beta = self._calculate_betas(dim, l_mat, rho)
            solution = self._generate_solution(beta, kernel, alpha, points, pixels, intrinsics)
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
            solutions = self._optimization_step(solutions, kernel, bases, alpha, points, pixels, intrinsics)
        return solutions['pose']

    def _generate_solution(self, beta, kernel, alpha, points, pixels, intrinsics, request_error=True):
        solution = dict()

        ctrl_pts_c = bmv(kernel, beta)
        ctrl_pts_c, points_c, sc = self._compute_norm_sign_scaling_factor(ctrl_pts_c, alpha, points)
        pose = self._get_se3(points, points_c)
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

    def _optimization_step(self, solutions, kernel, ctrl_pts_w, alpha, points, pixels, intrinsics):
        """
        Args:
            solutions (dict): a dict of solutions
            kernel (Tensor): kernel matrix, shape (batch, n, 4)
            ctrl_pts_w (Tensor): control points in the world coordinate, shape (batch, 4, 3)
            alpha (Tensor): alpha, shape (batch, n, 4)
            points (Tensor): 3D object points, shape (batch, n, 3)
            pixels (Tensor): 2D image points, shape (batch, n, 2)
            intrinsics (Tensor): camera intrinsics, shape (batch, 3, 3)

        Returns:
            None. This function will update the solutions in place.
        """
        objective = BetasOptimizationObjective(solutions['beta'] * solutions['scale'].unsqueeze(-1))
        gn = self.optimizer(objective)
        scheduler = StopOnPlateau(gn, steps=10, patience=3, verbose=False)
        scheduler.optimize(input=(ctrl_pts_w, kernel))
        beta = objective.betas.data

        solution = self._generate_solution(beta, kernel, alpha, points, pixels, intrinsics, request_error=False)
        return solution

    @staticmethod
    def _naive_basis(points):
        # Select 4 naive points, 3 unit bases and 1 origin point.
        controls = torch.zeros_like(points[..., :4, :])
        controls.diagonal(dim1=-2, dim2=-1).fill_(1)
        return controls

    @staticmethod
    def _svd_basis(points):
        # Select 4 control points with SVD
        center = points.mean(dim=-2, keepdim=True)
        translated = points - center
        u, s, vh = torch.linalg.svd(translated.mT @ translated)
        controls = center + s.sqrt().unsqueeze(-1) * vh.mT
        return torch.cat([center, controls], dim=-2)

    @staticmethod
    def _compute_alphas(points, bases):
        # Compute weights (..., N, 4) corresponded to bases (..., N, 4)
        # for points (..., N, 3). Check equation 1 in paper for details.
        points, bases = cart2homo(points), cart2homo(bases)
        return torch.linalg.solve(A=bases, B=points, left=False)

    @staticmethod
    def _calculate_kernel(pixels, alpha, intrinsics, least=4):
        # Construct M matrix and find its eigenvectors with the least eigenvalues
        # Check equation 7 in paper for more details.
        # pixels (..., point, 2); alpha (..., point, 4); intrinsics (..., 3, 3)
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
                         O, a3 * fv, a3 * (v0 - v)], dim=-1).reshape(*batch, point * 2, 12)
        eigenvalues, eigenvectors = torch.linalg.eig(M.mT @ M)
        eigenvalues, eigenvectors = eigenvalues.real, eigenvectors.real
        _, index = eigenvalues.topk(k=least, largest=False, sorted=True) # (batch, 4)
        index = index.flip(dims=[-1]).unsqueeze(-2).tile((1,)*len(batch)+(12, 1))
        return torch.gather(eigenvectors, dim=-1, index=index) # (batch, 12, 4)

    def _build_l(self, kernel_bases):
        """Given the kernel of m, compute the L matrix. Check [source]
        (https://github.com/cvlab-epfl/EPnP/blob/5abc3cfa76e8e92e5a8f4be0370bbe7da246065e/cpp/epnp.cpp#L478)
        for more details. Inputs are batched.

        Args:
            kernel_bases (torch.Tensor): kernel of m, shape (..., 12, 4)

        Returns:
            torch.Tensor: L, shape (..., 6, 10)
        """
        batch = kernel_bases.shape[:-2]
        kernel_bases = kernel_bases.mT  # shape (batch, 4, 12)
        # calculate the pairwise distance matrix within bases
        diff = kernel_bases.reshape(*batch, 4, 1, 4, 3) - kernel_bases.reshape(*batch, 4, 4, 1, 3)
        diff = diff.flatten(start_dim=-3, end_dim=-2)  # shape (batch, 4, 16, 3)
        # six_indices are (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3) before flatten
        dv = diff[..., self.six_indices, :]  # shape (batch, 4, 6, 3)

        # generate l
        dot_products = torch.sum(
            dv[..., self.ten_indices_pair[0], :, :] * dv[..., self.ten_indices_pair[1], :, :], dim=-1)
        dot_products = dot_products * self.multiply_mask.reshape((1,) * len(batch) + (10, 1))
        return dot_products.mT  # shape (batch, 6, 10)

    def _build_rho(self, bases):
        """Given the coordinates of control points, compute the rho vector. Check [source]
        (https://github.com/cvlab-epfl/EPnP/blob/5abc3cfa76e8e92e5a8f4be0370bbe7da246065e/cpp/epnp.cpp#L520)
        for more details.
        Inputs are batched.

        Args:
            cont_pts_w (torch.Tensor): coordinates of control points, shape (..., 4, 3)
        Returns:
            torch.Tensor: rho, shape (..., 6, 1)
        """
        dist = bases[..., self.six_indices_pair[0], :] - bases[..., self.six_indices_pair[1], :]
        return torch.sum(dist ** 2, dim=-1)  # l2 norm

    @staticmethod
    def _calculate_betas(dim, l_mat, rho):
        """Given the L matrix and rho vector, compute the beta vector. Check equation 10 - 14 in paper for more details.
        Inputs are batched.

        Args:
            dim (int): dimension of the problem, 1, 2, or 3
            l_mat (torch.Tensor): L, shape (..., 6, 10)
            rho (torch.Tensor): rho, shape (..., 6)
        Returns:
            torch.Tensor: beta, shape (..., 4)
        """
        batch = l_mat.shape[:-2]
        if dim == 1:
            betas = torch.zeros(*batch, 4, device=l_mat.device, dtype=l_mat.dtype)
            betas[..., -1] = 1
            return betas

        elif dim == 2:
            l_mat = l_mat[..., (5, 8, 9)]  # matched with matlab code
            betas = bmv(torch.linalg.pinv(l_mat), rho)  # shape: (b, 3)
            beta1 = betas[..., 0].abs().sqrt()
            beta2 = betas[..., 2].abs().sqrt() * betas[..., 1].sign() * betas[..., 0].sign()
            zeros = torch.zeros_like(beta1)
            return torch.stack([zeros, zeros, beta1, beta2], dim=-1)
    
        elif dim == 3:
            l_mat = l_mat[..., (2, 4, 7, 5, 8, 9)]  # matched with matlab code
            betas = torch.linalg.solve(l_mat, rho.unsqueeze(-1)).squeeze(-1)  # shape: (b, 6)
            beta1 = betas[..., 0].abs().sqrt()
            zeros = torch.zeros_like(beta1)
            beta2 = betas[..., 3].abs().sqrt() * betas[..., 1].sign() * betas[..., 0].sign()
            beta3 = betas[..., 5].abs().sqrt() * betas[..., 2].sign() * betas[..., 0].sign()
            return torch.stack([zeros, beta1, beta2, beta3], dim=-1)

        elif dim == 4:
            betas = bmv(torch.linalg.pinv(l_mat), rho)  # shape: (b, 10)
            beta4 = betas[..., 0].abs().sqrt()
            beta3 = betas[..., 2].abs().sqrt() * betas[..., 1].sign() * betas[..., 0].sign()
            beta2 = betas[..., 5].abs().sqrt() * betas[..., 3].sign() * betas[..., 0].sign()
            beta1 = betas[..., 9].abs().sqrt() * betas[..., 6].sign() * betas[..., 0].sign()
            return torch.stack([beta1, beta2, beta3, beta4], dim=-1)

    @staticmethod
    def _compute_norm_sign_scaling_factor(xc, alphas, points):
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
        batch = xc.shape[:-1]
        # Calculate the control points and object points in the camera coordinates
        ctrl_pts_c = xc.reshape((*batch, 4, 3))
        points_c = alphas @ ctrl_pts_c

        # Calculate the distance of the reference points in the world coordinates
        points_w_centered = points - points.mean(dim=-2, keepdim=True)
        dist_w = torch.linalg.norm(points_w_centered, dim=-1)

        # Calculate the distance of the reference points in the camera coordinates
        points_c_centered = points_c - points_c.mean(dim=-2, keepdim=True)
        dist_c = torch.linalg.norm(points_c_centered, dim=-1)

        # calculate the scaling factors
        # below are batched vector dot product
        sc = 1 / torch.linalg.vecdot(dist_c, dist_c) * torch.linalg.vecdot(dist_c, dist_w)

        # Update the control points and the object points in the camera coordinates based on the scaling factors
        ctrl_pts_c = ctrl_pts_c * sc[..., None, None]
        points_c = alphas.matmul(ctrl_pts_c)

        # Update the control points and the object points in the camera coordinates based on the sign
        neg_z_mask = torch.any(points_c[..., 2] < 0, dim=-1)  # (N, )

        # for batched data and non-batched data
        negate_switch = torch.ones(batch, dtype=points.dtype, device=points.device)
        negate_switch[neg_z_mask] = negate_switch[neg_z_mask] * -1
        points_c = points_c * negate_switch.reshape(*batch, 1, 1)
        sc = sc * negate_switch
        return ctrl_pts_c, points_c, sc

    @staticmethod
    def _get_se3(pts_w, pts_c):
        """
        Get the rotation matrix and translation vector based on the object points in world coordinate and camera
        coordinate.

        Args:
            pts_w: The object points in world coordinate. The shape is (..., N, 3).
            pts_c: The object points in camera coordinate. The shape is (..., N, 3).
        Returns:
            ``LieTensor``: estimated pose (``SE3type``) for the camera.
        """
        # Get the centered points
        center_w = pts_w.mean(dim=-2)
        pts_w = pts_w - center_w.unsqueeze(-2)
        center_c = pts_c.mean(dim=-2)
        pts_c = pts_c - center_c.unsqueeze(-2)

        # Calculate the rotation matrix
        m = torch.matmul(pts_c[..., :, None], pts_w[..., None, :])
        m = m.sum(dim=-3)  # along the point dimension

        u, s, vh = torch.svd(m)
        rot = u.matmul(vh.mT)

        # if det(R) < 0, make it positive
        negate_mask = torch.linalg.det(rot) < 0
        rot[negate_mask] = -rot[negate_mask]

        # Calculate the translation vector based on the rotation matrix and the equation
        t = center_c - bmv(rot, center_w)
        rt = torch.cat((rot, t.unsqueeze(-1)), dim=-1)

        return mat2SE3(rt)
