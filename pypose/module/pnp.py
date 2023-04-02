import torch
from .. import mat2SE3
from .. import bmv, bvv
from .camera import Camera
from ..basics import cart2homo
from ..optim import GaussNewton
from torch import broadcast_shapes
from ..optim.scheduler import StopOnPlateau
from ..optim.solver import Cholesky, PINV, LSTSQ


class BetasObjective(torch.nn.Module):
    # Optimize the betas according to the objectives in the ePnP paper.
    # For the details, please refer to equation 15.
    def __init__(self, betas):
        super().__init__()
        self.betas = torch.nn.Parameter(betas)

    def forward(self, ctrl_pts_w, nullv):
        """
        Args:
            ctrl_pts_w: The control points in world coordinate. The shape is (B, 4, 3).
            kernel_bases: The kernel bases. The shape is (B, 16, 4).

        Returns:
            torch.Tensor: The loss. The shape is (B, ).
        """
        batch = nullv.shape[:-2]
        # calculate the control points in camera coordinate
        ctrl_pts_c = bmv(nullv.mT, self.betas)
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

    def __init__(self, intrinsics=None, refine=True):
        super().__init__()
        self.refine = refine
        self.solver = LSTSQ()
        if intrinsics is not None:
            self.register_buffer('intrinsics', intrinsics)

    def forward(self, points, pixels, intrinsics=None):
        r'''
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
        '''
        intrinsics = self.intrinsics if intrinsics is None else intrinsics
        batch = broadcast_shapes(points.shape[:-2], pixels.shape[:-2], intrinsics.shape[:-2])

        # Select naive and calculate alpha in the world coordinate
        bases = self._svd_basis(points)
        alpha = self._compute_alphas(points, bases)
        nullv = self._compute_nullv(pixels, alpha, intrinsics)
        l_mat, rho = self._build_lrho(nullv, bases)

        solution_keys = ['error', 'pose', 'ctrl_pts_c', 'points_c', 'beta', 'scale']
        solutions = {key: [] for key in solution_keys}
        for dim in range(1, 5):  # nullv space dimension of 4 is unstable
            beta = self._calculate_beta(dim, l_mat, rho)
            solution = self._generate_solution(beta, nullv, alpha, points, pixels, intrinsics)
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

        if self.refine:
            beta = self._refine(solutions['beta'] * solutions['scale'].unsqueeze(-1), nullv, bases)
            solutions = self._generate_solution(beta, nullv, alpha, points, pixels, intrinsics)

        return solutions['pose']

    def _generate_solution(self, beta, nullv, alpha, points, pixels, intrinsics, request_error=True):
        solution = dict()

        ctrl_pts_c = bmv(nullv.mT, beta)
        ctrl_pts_c, points_c, sc = self._compute_norm_sign_scaling_factor(ctrl_pts_c, alpha, points)
        pose = self._get_se3(points, points_c)
        if request_error:
            camera = Camera(pose, intrinsics)
            error = camera.reprojection_error(points, pixels)
            solution['error'] = error

        # save the solution
        solution['pose'] = pose
        solution['ctrl_pts_c'] = ctrl_pts_c
        solution['points_c'] = points_c
        solution['beta'] = beta
        solution['scale'] = sc

        return solution

    def _refine(self, beta, nullv, bases):
        """
        Args:
            solutions (dict): a dict of solutions
            nullv (Tensor): null vectors of M matrix, shape (batch, n, 4)
            bases (Tensor): control points in the world coordinate, shape (batch, 4, 3)
        """
        objective = BetasObjective(beta)
        optim = GaussNewton(objective, solver=LSTSQ())
        scheduler = StopOnPlateau(optim, steps=10, patience=3)
        scheduler.optimize(input=(bases, nullv))
        return objective.betas

    @staticmethod
    def _svd_basis(points):
        # Select 4 virtual control points with SVD
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
    def _compute_nullv(pixels, alpha, intrinsics, least=4):
        # Construct M matrix and find its null eigenvectors with the least eigenvalues
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
                         O, a3 * fv, a3 * (v0 - v)], dim=-1).view(*batch, point * 2, 12)
        eigenvalues, eigenvectors = torch.linalg.eig(M.mT @ M)
        eigenvalues, eigenvectors = eigenvalues.real, eigenvectors.real
        _, index = eigenvalues.topk(k=least, largest=False, sorted=True) # (batch, 4)
        index = index.flip(dims=[-1]).unsqueeze(-2).tile((1,)*len(batch)+(12, 1))
        return torch.gather(eigenvectors, dim=-1, index=index).mT # (batch, 4, 12)

    @staticmethod
    def _build_lrho(nullv, bases):
        # prepare l_mat and rho to compute beta
        nullv = nullv.unflatten(dim=-1, sizes=(4, 3))
        i = (1, 2, 3, 2, 3, 3)
        j = (0, 0, 0, 1, 1, 2)
        dv = nullv[..., i, :] - nullv[..., j, :]
        a = (0, 0, 1, 0, 1, 2, 0, 1, 2, 3)
        b = (0, 1, 1, 2, 2, 2, 3, 3, 3, 3)
        dp = (dv[..., a, :, :] * dv[..., b, :, :]).sum(dim=-1)
        m = torch.tensor([1, 2, 1, 2, 2, 1, 2, 2, 2, 1], device=dp.device, dtype=dp.dtype)
        return dp.mT * m, (bases[..., i, :] - bases[..., j, :]).pow(2).sum(-1)

    def _calculate_beta(self, dim, l_mat, rho):
        # Given the L matrix and rho vector, compute the beta vector.
        # Check Eq 10 - 14 in paper.
        # l_mat (..., 6, 10); rho (..., 6); beta (..., 4)
        beta = torch.zeros_like(rho[...,:4])
        if dim == 1:
            beta[..., -1] = 1
        elif dim == 2:
            L = l_mat[..., (5, 8, 9)]
            res = self.solver(L, rho) # (b, 3)
            beta[..., 2] = res[..., 0].abs().sqrt()
            beta[..., 3] = res[..., 2].abs().sqrt() * res[..., 1].sign() * res[..., 0].sign()
        elif dim == 3:
            L = l_mat[..., (2, 4, 7, 5, 8, 9)]
            res = self.solver(L, rho) # (b, 6)
            beta[..., 1] = res[..., 0].abs().sqrt()
            beta[..., 2] = res[..., 3].abs().sqrt() * res[..., 1].sign() * res[..., 0].sign()
            beta[..., 3] = res[..., 5].abs().sqrt() * res[..., 2].sign() * res[..., 0].sign()
        elif dim == 4:
            res = self.solver(l_mat, rho) # (b, 10)
            beta[..., 0] = res[..., 9].abs().sqrt() * res[..., 6].sign() * res[..., 0].sign()
            beta[..., 1] = res[..., 5].abs().sqrt() * res[..., 3].sign() * res[..., 0].sign()
            beta[..., 2] = res[..., 2].abs().sqrt() * res[..., 1].sign() * res[..., 0].sign()
            beta[..., 3] = res[..., 0].abs().sqrt()
        return beta

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
        # Get transform for two associated batched point sets.
        Cw = pts_w.mean(dim=-2, keepdim=True)
        Pw = pts_w - Cw
        Cc = pts_c.mean(dim=-2, keepdim=True)
        Pc = pts_c - Cc
        M = bvv(Pc, Pw).sum(dim=-3)
        U, S, Vh = torch.linalg.svd(M)
        R = U @ Vh
        # mirror improper rotation that det(R) = -1
        mask = (R.det() + 1).abs() < 1e-6
        R[mask] = - R[mask]
        t = Cc.mT - R @ Cw.mT
        T = torch.cat((R, t), dim=-1)
        return mat2SE3(T, check=False)
