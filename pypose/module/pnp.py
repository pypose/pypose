import torch
import pypose
from pypose.optim import GaussNewton


class EPnP(torch.nn.Module):
    r"""
        EPnP Solver - a non-iterative O(n) solution to the PnP problem.

        Args:
            optimizer (Optional[torch.optim.Optimizer]): Optimizer to refine the solution. Set to None to disable refinement.
            naive_ctrl_pts (bool): Use naive control points selection method. Reco

        Examples:
            >>> import torch
            >>> import pypose
            >>> # create some random test sample for a single camera
            >>> pose = pypose.SE3([ 0.0000, -8.0000,  0.0000,  0.0000, -0.3827,  0.0000,  0.9239])
            >>> f = 2
            >>> img_size = (7, 7)
            >>> projection_matrix = torch.tensor([[f, 0, img_size[0] / 2,], [0, f, img_size[1] / 2,], [0, 0, 1, ]])
            >>> # some random points in the view
            >>> pts_c = torch.tensor([[2., 0., 2.], [1., 0., 2.], [0., 1., 1.], [0., 0., 1.], [5., 5., 3.]])
            >>> pixels = (pts_c @ projection_matrix.T)[:, :2] / (pts_c @ projection_matrix.T)[:, 2:]
            >>> pixels
            tensor([[5.5000, 3.5000],
                    [4.5000, 3.5000],
                    [3.5000, 5.5000],
                    [3.5000, 3.5000],
                    [6.8333, 6.8333]])
            >>> # transform the points to world coordinate
            >>> # solve the PnP problem to find the camera pose
            >>> pts_w = pose.Inv().Act(pts_c)
            >>> # solve the PnP problem
            >>> epnp = pypose.module.EPnP()
            >>> # when input is not batched, remember to add a batch dimension
            >>> pose = epnp(pts_w[None], pixels[None], projection_matrix[None])
            >>> pose
            SE3Type LieTensor:
            LieTensor([[ 5.4955e-05, -8.0000e+00, -2.7895e-05,  6.8488e-06, -3.8270e-01,
                         3.2812e-06,  9.2387e-01]])

        Note:
            The implementation is based on the paper:
            * Francesc Moreno-Noguer, Vincent Lepetit, and Pascal Fua, `Accurate Non-Iterative O(n) Solution to the PnP
              Problem. <https://github.com/cvlab-epfl/EPnP>`_,
              In Proceedings of ICCV, 2007.

    """

    class BetasOptimizationObjective(torch.nn.Module):
        """
        Optimize the betas according to the objectives in the paper.
        For the details, please refer to equation 15.
        """

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
            batch_shape = kernel_bases.shape[:-2]
            # calculate the control points in camera coordinate
            ctrl_pts_c = pypose.bmv(kernel_bases, self.betas)
            diff_c = ctrl_pts_c.reshape(*batch_shape, 1, 4, 3) - ctrl_pts_c.reshape(*batch_shape, 4, 1, 3)
            diff_c = diff_c.reshape(*batch_shape, 16, 3)
            diff_c = torch.sum(diff_c ** 2, dim=-1)

            # calculate the distance between control points in world coordinate
            diff_w = ctrl_pts_w.reshape(*batch_shape, 1, 4, 3) - ctrl_pts_w.reshape(*batch_shape, 4, 1, 3)
            diff_w = diff_w.reshape(*batch_shape, 16, 3)
            diff_w = torch.sum(diff_w ** 2, dim=-1)

            error = (diff_w - diff_c).abs().mean(dim=-1)

            return error

    def __init__(self, naive_ctrl_pts=False, optimizer=GaussNewton):
        super().__init__()
        self.naive_ctrl_pts = naive_ctrl_pts
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
            points (Tensor): 3D object points in the world coordinates, shape (batch_size, n, 3)
            pixels (Tensor): 2D image points, which are the projection of object points, with shape (batch_size, n, 2)
            intrinsics (Tensor): camera intrinsics, shape (batch_size, 3, 3)
        Returns:
            lietensor.SE3Type: estimated pose for the camera
        """
        # shape checking
        batch_shape = points.shape[:-2]
        assert pixels.shape[:-2] == batch_shape
        assert intrinsics.shape[:-2] == batch_shape

        # Select four control points and calculate alpha (in the world coordinate)
        if self.naive_ctrl_pts:
            ctrl_pts_w = self.naive_control_points(points)
        else:
            ctrl_pts_w = self.select_control_points(points)
        alpha = self.compute_alphas(points, ctrl_pts_w)

        # Using camera projection equation for all the points pairs to get the matrix M
        m = self.build_m(pixels, alpha, intrinsics)

        kernel_m = self.calculate_kernel(m)[..., [3, 2, 1, 0]]  # to be consistent with the matlab code

        l_mat = self.build_l(kernel_m)
        rho = self.build_rho(ctrl_pts_w)

        solution_keys = ['error', 'pose', 'ctrl_pts_c', 'points_c', 'beta', 'scale']
        solutions = {key: [] for key in solution_keys}
        for dim in range(1, 4):  # kernel space dimension of 4 is unstable
            beta = self.calculate_betas(dim, l_mat, rho)
            solution = self.generate_solution(beta, kernel_m, alpha, points, pixels, intrinsics)
            for key in solution_keys:
                solutions[key].append(solution[key])

        # stack the results
        for key in solutions.keys():
            solutions[key] = torch.stack(solutions[key], dim=len(batch_shape))
        best_error, best_idx = torch.min(solutions['error'], dim=len(batch_shape))
        for key in solutions.keys():
            # retrieve the best solution using gather
            best_idx_ = best_idx.reshape(best_idx.shape + (1,) * (solutions[key].dim() - len(batch_shape)))
            best_idx_ = best_idx_.tile((1,) * (len(batch_shape) + 1) + solutions[key].shape[len(batch_shape) + 1:])
            solutions[key] = torch.gather(solutions[key], len(batch_shape), best_idx_)
            solutions[key] = solutions[key].squeeze(len(batch_shape))

        if self.optimizer is not None:
            solutions = self.optimization_step(solutions, kernel_m, ctrl_pts_w, alpha, points, pixels, intrinsics)
        return solutions['pose']

    def generate_solution(self, beta, kernel_m, alpha, points, pixels, intrinsics, request_error=True):
        solution = dict()

        ctrl_pts_c = pypose.bmv(kernel_m, beta)
        ctrl_pts_c, points_c, sc = self.compute_norm_sign_scaling_factor(ctrl_pts_c, alpha, points)
        pose = self.get_se3(points, points_c)
        if request_error:
            perspective_camera = pypose.module.PerspectiveCameras(pose, intrinsics)
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
        objective = self.BetasOptimizationObjective(solutions['beta'] * solutions['scale'].unsqueeze(-1))
        gn = self.optimizer(objective)
        scheduler = pypose.optim.scheduler.StopOnPlateau(gn, steps=10, patience=3, verbose=False)
        scheduler.optimize(input=(ctrl_pts_w, kernel_m))
        beta = objective.betas.data

        solution = self.generate_solution(beta, kernel_m, alpha, points, pixels, intrinsics, request_error=False)
        return solution

    @staticmethod
    def naive_control_points(points):
        """
        Select four control points, used to express world coordinates of the object points. This is a naive
        implementation that corresponds to the original paper.

        Args:
            points: 3D object points, shape (..., n, 3)
        Returns:
            control points, shape (..., 4, 3)
        """
        batch_shape = points.shape[:-2]
        control_pts = torch.eye(3, dtype=points.dtype, device=points.device)
        # last control point is the origin
        control_pts = torch.cat((control_pts, torch.zeros(1, 3, dtype=points.dtype, device=points.device)), dim=0)
        control_pts = control_pts.reshape((1,) * len(batch_shape) + (4, 3)).repeat(*batch_shape, 1, 1)
        return control_pts

    @staticmethod
    def select_control_points(points):
        """
        Select four control points, used to express world coordinates of the object points

        Args:
            points: 3D object points, shape (..., n, 3)
        Returns:
            control points, shape (..., 4, 3)
        """
        # Select the center of mass to be the first control point
        center = points.mean(axis=-2)

        # Use distance to center to select the other three control points
        # svd
        centered_points = points - center.unsqueeze(-2)  # center the object points, 1 is for broadcasting
        u, s, vh = torch.linalg.svd(torch.bmm(centered_points.mT, centered_points), full_matrices=True)

        # produce points TODO: change to batch implementation
        res = [center, ]
        for i in range(3):
            another_pt = center + torch.sqrt(s[..., i, None]) * vh[..., i]
            res.append(another_pt)

        return torch.stack(res, dim=-2)

    @staticmethod
    def compute_alphas(points, ctrl_pts_w):
        """Given the object points and the control points in the world coordinate, compute the alphas,
        which are a set of coefficients corresponded of control points for each object point. Check equation 1 in paper
        for more details.
        Inputs are batched.

        Args:
            points (torch.Tensor): object points in the world coordinate, shape (..., num_pts, 3)
            ctrl_pts_w (torch.Tensor): control points in the world coordinate, shape (..., 4, 3)
        Returns:
            torch.Tensor: alphas, shape (..., num_pts, 4)
        """
        batch_shape = points.shape[:-2]
        num_pts = points.shape[1]
        batched_ones = torch.ones((*batch_shape, num_pts, 1), dtype=points.dtype, device=points.device)
        # concatenate object points with ones
        points = torch.cat((points, batched_ones), dim=-1)
        # concatenate control points with ones
        batched_ones = torch.ones((*batch_shape, 4, 1), dtype=ctrl_pts_w.dtype, device=ctrl_pts_w.device)
        ctrl_pts_w = torch.cat((ctrl_pts_w, batched_ones), dim=-1)

        alpha = torch.linalg.solve(ctrl_pts_w, points, left=False)  # General method
        return alpha

    @staticmethod
    def build_m(pixels, alpha, intrinsics):
        """Given the image points, alphas and intrinsics, compute the m matrix, which is the matrix of the coefficients
        of the image points. Check equation 7 in paper for more details.
        Inputs are batched.

        Args:
            pixels (torch.Tensor): image points, shape (..., num_pts, 2)
            alpha (torch.Tensor): alphas, shape (..., num_pts, 4)
            intrinsics (torch.Tensor): intrinsics, shape (..., 3, 3)
        Returns:
            torch.Tensor: m, shape (..., num_pts * 2, 12)
        """
        batch_shape = pixels.shape[:-2]
        num_pts = pixels.shape[-2]

        # extract elements of the intrinsic matrix in batch
        fu, fv, u0, v0 = (intrinsics[..., 0, 0, None],
                          intrinsics[..., 1, 1, None],
                          intrinsics[..., 0, 2, None],
                          intrinsics[..., 1, 2, None])
        # extract elements of the image points in batch
        ui, vi = pixels[..., 0], pixels[..., 1]
        # extract elements of the alphas in batch
        a1, a2, a3, a4 = alpha[..., 0], alpha[..., 1], alpha[..., 2], alpha[..., 3]
        # build zero tensor, with a batch and point dimension
        zeros = torch.zeros_like(a1)
        # build by order the last dimension of m, shape (batch_size, num_pts, 24)
        m_ = [a1 * fu, zeros, a1 * (u0 - ui),
              a2 * fu, zeros, a2 * (u0 - ui),
              a3 * fu, zeros, a3 * (u0 - ui),
              a4 * fu, zeros, a4 * (u0 - ui),
              zeros, a1 * fv, a1 * (v0 - vi),
              zeros, a2 * fv, a2 * (v0 - vi),
              zeros, a3 * fv, a3 * (v0 - vi),
              zeros, a4 * fv, a4 * (v0 - vi)]
        m = torch.stack(m_, dim=-1)

        # match dimension of m with paper, with shape (batch_size, num_pts * 2, 12)
        m = m.reshape(*batch_shape, num_pts * 2, 12)
        return m

    @staticmethod
    def calculate_kernel(m, top=4):
        """Given the m matrix, compute the kernel of it. Check equation 8 in paper for more details.
        Inputs are batched.

        Args:
            m (torch.Tensor): m, shape (..., num_pts * 2, 12)
            top (int, optional): number of top eigen vectors to take. Defaults to 4.
        Returns:
            torch.Tensor: kernel, shape (..., 12, top)
        """
        batch_shape = m.shape[:-2]
        # find null space of M
        eigenvalues, eigenvectors = torch.linalg.eig(torch.matmul(m.mT, m))
        # take the real part
        eigenvalues = eigenvalues.real
        eigenvectors = eigenvectors.real
        # sort by eigenvalues (ascending)
        eig_indices = eigenvalues.argsort()
        # take the first 4 eigenvectors, shape (batch_size, 12, 4)
        kernel_bases = torch.gather(eigenvectors, -1,
                                    eig_indices[..., :top].unsqueeze(-2).tile((1,) * len(batch_shape) + (12, 1)))

        return kernel_bases

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
            betas_ = pypose.bmv(torch.linalg.pinv(l_mat), rho)  # shape: (b, 3)
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
            betas_ = pypose.bmv(torch.linalg.pinv(l_mat), rho)  # shape: (b, 10)
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
        points_c = torch.bmm(alphas, ctrl_pts_c)

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
        t = center_c - pypose.bmv(rot, center_w)
        rt = torch.cat((rot, t.unsqueeze(-1)), dim=-1)
        pose = pypose.mat2SE3(rt)

        return pose

