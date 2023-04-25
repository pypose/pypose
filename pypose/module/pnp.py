import torch
from torch.linalg import vecdot
from torch import broadcast_shapes

from .. import bmv
from ..optim import GaussNewton
from ..optim.solver import LSTSQ
from ..optim.scheduler import StopOnPlateau
from ..function import reprojerr, cart2homo, svdtf


class BetaObjective(torch.nn.Module):
    # Optimize the beta according to the objective in the ePnP paper.
    def __init__(self, beta):
        super().__init__()
        self.beta = torch.nn.Parameter(beta)
        self.i = (0, 0, 0, 1, 1, 2)
        self.j = (1, 2, 3, 2, 3, 3)

    def forward(self, base_w, nullv):
        # See Eq. 15 in the paper
        base_c = bmv(nullv.mT, self.beta).unflatten(dim=-1, sizes=(4, 3))
        dist_c = (base_c[..., self.i, :] - base_c[..., self.j, :]).norm(dim=-1)
        dist_w = (base_w[..., self.i, :] - base_w[..., self.j, :]).norm(dim=-1)
        return dist_w - dist_c


class EPnP(torch.nn.Module):
    r'''
    Batched EPnP Solver - a non-iterative :math:`\mathcal{O}(n)` solution to the
    Perspective-:math:`n`-Point (PnP) problem for :math:`n \geq 4`.

    Args:
        intrinsics (``torch.Tensor``, optional): The camera intrinsics.
            The shape is (..., 3, 3). Default: None
        refine (``bool``, optional): refine the solution with Gaussian-Newton optimizer.
            Default: ``True``.

    Assume each of the :math:`n` points in the world coordinate is :math:`p^w_i` and in
    camera coordinate is :math:`p^c_i`.
    They are represented by weighted sums of the four virtual control points,
    :math:`c^w_j` and :math:`c^c_j` in the world and camera coordinate, respectively.

    .. math::
        \begin{aligned}
            & p^w_i = \sum^4_{j=1}{\alpha_{ij}c^w_j} \\
            & p^c_i = \sum^4_{j=1}{\alpha_{ij}c^c_j} \\
            & \sum^4_{j=1}{\alpha_{ij}} = 1
        \end{aligned}

    Let the projection matrix be :math:`P = K[R|T]`, where :math:`K` is the camera
    intrinsics, :math:`R` is the rotation matrix and :math:`T` is the translation
    vector. Then we have

    .. math::
        \begin{aligned}
            s_i p^{\text{img}}_i &= K\sum^4_{j=1}{\alpha_{ij}c^c_j},
        \end{aligned}

    where :math:`p^{\text{img}}_i` is pixels in homogeneous form :math:`(u_i, v_i, 1)`,
    :math:`s_i` is the scale factor. Let the control point in camera coordinate
    represented by :math:`c^c_j = (x^c_j, y^c_j, z^c_j)`. Rearranging the projection
    equation yields two linear equations for each of the :math:`n` points:

    .. math::
        \begin{aligned}
            \sum^4_{j=1}{\alpha_{ij}f_xx^c_j + \alpha_{ij}(u_0 - u_i)z^c_j} &= 0 \\
            \sum^4_{j=1}{\alpha_{ij}f_yy^c_j + \alpha_{ij}(v_0 - v_i)z^c_j} &= 0
        \end{aligned}

    Assume :math:`\mathbf{x} = \begin{bmatrix} c^{c^T}_1 & c^{c^T}_2 & c^{c^T}_3 &
    c^{c^T}_4 \end{bmatrix}^T`, then the two equations form a system :math:`Mx = 0`
    considering all of the :math:`n` points. Its solution can be expressed as

    .. math::
        \begin{aligned}
            x &= \sum^4_{i=1}{\beta_iv_i},
        \end{aligned}

    where :math:`v_i` is the null vectors of matrix :math:`M^T M` corresponding to its
    least 4 eigenvalues.

    The final step involves calculating the coefficients :math:`\beta_i`. Optionally, the
    Gauss-Newton algorithm can be used to refine the solution of :math:`\beta_i`.

    Example:
        >>> import torch, pypose as pp
        >>> f, (H, W) = 2, (9, 9) # focal length and image height, width
        >>> intrinsics = torch.tensor([[f, 0, H / 2],
        ...                            [0, f, W / 2],
        ...                            [0, 0,   1  ]])
        >>> object = torch.tensor([[2., 0., 2.],
        ...                        [1., 0., 2.],
        ...                        [0., 1., 1.],
        ...                        [0., 0., 1.],
        ...                        [1., 0., 1.],
        ...                        [5., 5., 3.]])
        >>> pixels = pp.point2pixel(object, intrinsics)
        >>> pose = pp.SE3([ 0., -8,  0.,  0., -0.3827,  0.,  0.9239])
        >>> points = pose.Inv() @ object
        ...
        >>> epnp = pp.module.EPnP(intrinsics)
        >>> pose = epnp(points, pixels)
        SE3Type LieTensor:
        LieTensor([ 3.9816e-05, -8.0000e+00,  5.8174e-05, -3.3186e-06, -3.8271e-01,
                    3.6321e-06,  9.2387e-01])

    Warning:
        Currently this module only supports batched rectified camera intrinsics, which can
        be defined in the form:

        .. math::
            K = \begin{pmatrix}
                    f_x &   0 & c_x \\
                    0   & f_y & c_y \\
                    0   &   0 &   1
                \end{pmatrix}

        The full form of camera intrinsics will be supported in a future release.

    Note:
        The implementation is based on the paper

        * Francesc Moreno-Noguer, Vincent Lepetit, and Pascal Fua, `EPnP: An Accurate O(n)
          Solution to the PnP Problem <https://doi.org/10.1007/s11263-008-0152-6>`_,
          International Journal of Computer Vision (IJCV), 2009.
    '''
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
                Shape (..., N, 3)
            pixels (``torch.Tensor``): 2D image points, which are the projection of
                object points. Shape (..., N, 2)
            intrinsics (torch.Tensor, optional): camera intrinsics. Shape (..., 3, 3).
                Setting it to any non-``None`` value will override the default intrinsics
                kept in the module.

        Returns:
            ``LieTensor``: estimated pose (``SE3type``) for the camera.
        '''
        assert pixels.size(-2) == points.size(-2) >= 4, \
            "Number of points/pixels cannot be smaller than 4."
        intrinsics = self.intrinsics if intrinsics is None else intrinsics
        broadcast_shapes(points.shape[:-2], pixels.shape[:-2], intrinsics.shape[:-2])

        # Select naive and calculate alpha in the world coordinate
        bases = self._svd_basis(points)
        alpha = self._compute_alpha(points, bases)
        nullv = self._compute_nullv(pixels, alpha, intrinsics)
        l_mat, rho = self._compute_lrho(nullv, bases)
        betas = self._compute_betas(l_mat, rho)
        poses, scales = self._compute_solution(betas, nullv, alpha, points)
        errors = reprojerr(points, pixels, intrinsics, poses)
        pose, beta, scale = self._best_solution(errors, poses, betas, scales)

        if self.refine:
            beta = self._refine(beta * scale, nullv, bases)
            pose, scale = self._compute_solution(beta, nullv, alpha, points)

        return pose

    def _compute_solution(self, beta, nullv, alpha, points):
        bases = bmv(nullv.mT, beta)
        bases, transp, scale = self._compute_scale(bases, alpha, points)
        pose = svdtf(points, transp)
        return pose, scale

    @staticmethod
    def _best_solution(errors, poses, betas, scales):
        _, idx = torch.min(errors.mean(dim=-1, keepdim=True), dim=0, keepdim=True)
        pose = poses.gather(0, index=idx.tile(poses.size(-1))).squeeze(0)
        beta = betas.gather(0, index=idx.tile(betas.size(-1))).squeeze(0)
        scale = scales.gather(0, index=idx).squeeze(0)
        return pose, beta, scale

    @staticmethod
    def _refine(beta, nullv, bases):
        # Refine beta according to Eq 15 in the paper
        model = BetaObjective(beta)
        optim = GaussNewton(model, solver=LSTSQ())
        scheduler = StopOnPlateau(optim, steps=10, patience=3)
        scheduler.optimize(input=(bases, nullv))
        # Retain the grad of initial beta after optimization.
        return beta + (model.beta - beta).detach()

    @staticmethod
    def _svd_basis(points):
        # Select 4 virtual control points with SVD
        center = points.mean(dim=-2, keepdim=True)
        translated = points - center
        u, s, vh = torch.linalg.svd(translated.mT @ translated)
        controls = center + s.sqrt().unsqueeze(-1) * vh.mT
        return torch.cat([center, controls], dim=-2)

    @staticmethod
    def _compute_alpha(points, bases):
        # Compute weights (..., N, 4) corresponded to bases (..., N, 4)
        # for points (..., N, 3). Check equation 1 in paper for details.
        points, bases = cart2homo(points), cart2homo(bases)
        return torch.linalg.solve(A=bases, B=points, left=False)

    @staticmethod
    def _compute_nullv(pixels, alpha, intrinsics, least=4):
        # Construct M matrix and find its null eigenvectors with the least eigenvalues
        # Check equation 7 in paper for more details.
        # pixels (..., N, 2); alpha (..., N, 4); intrinsics (..., 3, 3)
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
        _, index = eigenvalues.topk(k=least, largest=False, sorted=True)  # (batch, 4)
        index = index.flip(dims=[-1]).unsqueeze(-2).tile((1,) * len(batch) + (12, 1))
        return torch.gather(eigenvectors, dim=-1, index=index).mT  # (batch, 4, 12)

    @staticmethod
    def _compute_lrho(nullv, bases):
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

    def _compute_betas(self, l_mat, rho):
        # Given the L matrix and rho vector, compute the betas vector.
        # Check Eq 10 - 14 in paper.
        # l_mat (..., 6, 10); rho (..., 6); betas (..., 4); return betas (4, ..., 4)
        betas = torch.zeros((4,)+rho.shape[:-1]+(4,), device=rho.device, dtype=rho.dtype)
        # dim == 1:
        betas[0, ..., -1] = 1
        # dim == 2:
        L = l_mat[..., (5, 8, 9)]
        S = self.solver(L, rho)  # (b, 3)
        betas[1, ..., 2] = S[..., 0].abs().sqrt()
        betas[1, ..., 3] = S[..., 2].abs().sqrt() * S[..., 1].sign() * S[..., 0].sign()
        # dim == 3:
        L = l_mat[..., (2, 4, 7, 5, 8, 9)]
        S = self.solver(L, rho)  # (b, 6)
        betas[2, ..., 1] = S[..., 0].abs().sqrt()
        betas[2, ..., 2] = S[..., 3].abs().sqrt() * S[..., 1].sign() * S[..., 0].sign()
        betas[2, ..., 3] = S[..., 5].abs().sqrt() * S[..., 2].sign() * S[..., 0].sign()
        # dim == 4:
        S = self.solver(l_mat, rho)  # (b, 10)
        betas[3, ..., 0] = S[..., 9].abs().sqrt() * S[..., 6].sign() * S[..., 0].sign()
        betas[3, ..., 1] = S[..., 5].abs().sqrt() * S[..., 3].sign() * S[..., 0].sign()
        betas[3, ..., 2] = S[..., 2].abs().sqrt() * S[..., 1].sign() * S[..., 0].sign()
        betas[3, ..., 3] = S[..., 0].abs().sqrt()
        return betas

    @staticmethod
    def _compute_scale(bases, alpha, points):
        # Compute the scaling factor and the sign of the scaling factor
        # input:  bases (4, ..., 12); alpha (..., N, 4); points (..., N, 3);
        # return: bases (4, ..., 4, 3); scalep (4, ..., N, 3); scale (4, ..., 1)
        bases = bases.unflatten(-1, (4, 3))
        transp = alpha @ bases # transformed points
        dw = (points - points.mean(dim=-2, keepdim=True)).norm(dim=-1)
        dc = (transp - transp.mean(dim=-2, keepdim=True)).norm(dim=-1)
        scale = vecdot(dc, dw) / vecdot(dc, dc)
        bases = bases * scale[..., None, None] # the real position
        scalep = alpha @ bases # scaled transformed points
        mask = torch.any(scalep[..., 2] < 0, dim=-1) # negate when z < 0
        sign = torch.ones_like(scale) - mask * 2     # 1 or -1
        scalep = sign[..., None, None] * scalep
        scale = (sign * scale).unsqueeze(-1)
        return bases, scalep, scale
