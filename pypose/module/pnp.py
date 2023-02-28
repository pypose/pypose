import torch
import functorch
from functools import partial
from functorch import vmap
from pypose.optim import GaussNewton


class EfficientPnP(torch.nn.Module):
    """
        EPnP Solver - a non-iterative O(n) solution to the PnP problem.
        as described in:

        Francesc Moreno-Noguer, Vincent Lepetit, Pascal Fua.
        Accurate Non-Iterative O(n) Solution to the PnP Problem.
        In Proceedings of ICCV, 2007.
        source: https://github.com/cvlab-epfl/EPnP

    """
    six_indices_pair = torch.tensor([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)], device='cpu').T
    six_indices = [(0 * 4 + 1), (0 * 4 + 2), (0 * 4 + 3), (1 * 4 + 2), (1 * 4 + 3), (2 * 4 + 3)]
    ten_indices_pair = torch.tensor([(0, 0),
                                     (0, 1), (1, 1),
                                     (0, 2), (1, 2), (2, 2),
                                     (0, 3), (1, 3), (2, 3), (3, 3)], device='cpu').T
    # equal mask [ True, False,  True, False, False,  True, False, False, False,  True]
    multiply_mask = torch.tensor([1., 2., 1., 2., 2., 1., 2., 2., 2., 1.], device='cpu')

    def __init__(self, gauss_newton=True, naive_ctrl_pts=False):
        """
        Args:
            distCoeff: Distortion matrix.

        Returns:
            Rt: Transform matrix include the rotation and the translation [R|t].
        """
        super().__init__()
        self.naive_ctrl_pts = naive_ctrl_pts
        self.gauss_newton = gauss_newton

    def forward(self, objPts, imgPts, intrinsics):
        # Select four control points and calculate alpha (in the world coordinate)
        if self.naive_ctrl_pts:
            contPts_w = self.naive_control_points(objPts)
        else:
            contPts_w = self.select_control_points(objPts)
        alpha = self.compute_alphas(objPts, contPts_w)

        # Using camera projection equation for all the points pairs to get the matrix M
        m = self.build_m(imgPts, alpha, intrinsics)

        kernel_m = self.calculate_kernel(m)[:, :, [3, 2, 1, 0]]  # to be consistent with the matlab code

        l = self.build_l(kernel_m)
        rho = self.build_rho(contPts_w)

        solution_keys = ['error', 'R', 't', 'contPts_c', 'objPts_c', 'beta', 'scale']
        solutions = {key: [] for key in solution_keys}
        for dim in range(1, 4):  # kernel space dimension of 4 is unstable
            beta = self.calculate_betas(dim, l, rho)

            contPts_c = torch.bmm(kernel_m, beta.unsqueeze(-1)).squeeze(-1)
            contPts_c, objPts_c, sc = self.compute_norm_sign_scaling_factor(contPts_c, alpha, objPts)
            r, t = self.get_rotation_translation(objPts, objPts_c)
            rt = torch.cat((r, t.unsqueeze(-1)), dim=-1)
            error = self.reprojection_error(objPts, imgPts, intrinsics, rt)

            # append the results
            solutions['error'].append(error)
            solutions['R'].append(r)
            solutions['t'].append(t)
            solutions['contPts_c'].append(contPts_c)
            solutions['objPts_c'].append(objPts_c)
            solutions['beta'].append(beta)
            solutions['scale'].append(sc)

        # stack the results
        for key in solutions.keys():
            solutions[key] = torch.stack(solutions[key], dim=1)
        best_error, best_idx = torch.min(solutions['error'], dim=1)
        for key in solutions.keys():
            # retrieve the best solution using gather
            best_idx_ = best_idx.reshape(best_idx.shape + (1,) * (solutions[key].dim() - 1))
            best_idx_ = best_idx_.tile((1, 1,) + solutions[key].shape[2:])
            solutions[key] = torch.gather(solutions[key], 1, best_idx_)
            solutions[key] = solutions[key].squeeze(1)

        if self.gauss_newton:
            self.guass_newton(solutions, kernel_m, contPts_w, alpha, objPts, imgPts, intrinsics)
        return solutions

    def guass_newton(self, solutions, kernel_m, contPts_w, alpha, objPts, imgPts, intrinsics):
        """
        Args:
            solutions: a dict of solutions
            kernel_m: kernel matrix, shape (batch_size, n, 4)
            contPts_w: control points in the world coordinate, shape (batch_size, 4, 3)
            alpha: alpha, shape (batch_size, n, 4)
            objPts: 3D object points, shape (batch_size, n, 3)
            imgPts: 2D image points, shape (batch_size, n, 2)
            intrinsics: camera intrinsics, shape (batch_size, 3, 3)
        Returns:
            None. This function will update the solutions in place.
        """
        objective = OptimizeBetas(solutions['beta'] * solutions['scale'].unsqueeze(-1))
        gn = GaussNewton(objective)
        errors = []
        best_error = solutions['error']
        for i in range(10):
            gn.step((contPts_w, kernel_m))
            beta = objective.betas.data
            contPts_c = torch.bmm(kernel_m, beta.unsqueeze(-1)).squeeze(-1)
            contPts_c, objPts_c, sc = self.compute_norm_sign_scaling_factor(contPts_c, alpha, objPts)
            r, t = self.get_rotation_translation(objPts, objPts_c)
            rt = torch.cat((r, t.unsqueeze(-1)), dim=-1)
            error = self.reprojection_error(objPts, imgPts, intrinsics, rt)
            errors.append(error)

            # update solutions
            update_mask = error < best_error
            solutions['error'][update_mask] = error[update_mask]
            solutions['R'][update_mask] = r[update_mask]
            solutions['t'][update_mask] = t[update_mask]
            solutions['contPts_c'][update_mask] = contPts_c[update_mask]
            solutions['objPts_c'][update_mask] = objPts_c[update_mask]
            solutions['beta'][update_mask] = beta[update_mask]
            solutions['scale'][update_mask] = sc[update_mask]

    @staticmethod
    def naive_control_points(objPts):
        """
        Select four control points, used to express world coordinates of the object points. This is a naive
        implementation that corresponds to the original paper.
        Args:
            objPts: 3D object points, shape (batch_size, n, 3)
        Returns:
            control points, shape (batch_size, 4, 3)
        """
        control_pts = torch.eye(3, dtype=objPts.dtype, device=objPts.device)
        # last control point is the origin
        control_pts = torch.cat((control_pts, torch.zeros(1, 3, dtype=objPts.dtype, device=objPts.device)), dim=0)
        control_pts = control_pts.unsqueeze(0).repeat(objPts.shape[0], 1, 1)
        return control_pts

    @staticmethod
    def select_control_points(objPts):
        """
        Select four control points, used to express world coordinates of the object points
        Args:
            objPts: 3D object points, shape (batch_size, n, 3)
        Returns:
            control points, shape (batch_size, 4, 3)
        """
        # Select the center of mass to be the first control point
        center = objPts.mean(axis=1)

        # Use distance to center to select the other three control points
        # svd
        objPts_w_cent = objPts - center.unsqueeze(1)  # center the object points, 1 is for boardcasting

        full_svd = vmap(partial(torch.linalg.svd, full_matrices=True))
        u, s, vh = full_svd(torch.bmm(objPts_w_cent.transpose(-1, -2), objPts_w_cent))

        # produce points TODO: change to batch implementation
        res = [center, ]
        for i in range(3):
            another_pt = center + torch.sqrt(s[:, i, None]) * vh[:, i]
            res.append(another_pt)

        return torch.stack(res, dim=1)

    @staticmethod
    def compute_alphas(objPts, contPts_w):
        """Given the object points and the control points in the world coordinate, compute the alphas, which are the coefficients corresponded of control points.
        Inputs are batched, with first dimension being the batch size. Check equation 1 in paper for more details.
        Args:
            objPts (torch.Tensor): object points in the world coordinate, shape (batch_size, num_pts, 3)
            contPts_w (torch.Tensor): control points in the world coordinate, shape (batch_size, 4, 3)
            linear_least_square (bool, optional): whether to use linear least square method to compute alphas. Defaults to False.
        Returns:
            torch.Tensor: alphas, shape (batch_size, num_pts, 4)
        """
        batch_size = objPts.shape[0]
        num_pts = objPts.shape[1]
        batched_ones = torch.ones((batch_size, num_pts, 1), dtype=objPts.dtype, device=objPts.device)
        # concatenate object points with ones
        objPts = torch.cat((objPts, batched_ones), dim=-1)
        # concatenate control points with ones
        batched_ones = torch.ones((batch_size, 4, 1), dtype=contPts_w.dtype, device=contPts_w.device)
        contPts_w = torch.cat((contPts_w, batched_ones), dim=-1)

        alpha = torch.linalg.solve(contPts_w, objPts, left=False)  # General method
        return alpha

    @staticmethod
    def build_m(imgPts, alpha, intrinsics):
        """Given the image points, alphas and intrinsics, compute the m matrix, which is the matrix of the coefficients
        of the image points. Check equation 7 in paper for more details.
        Inputs are batched, with first dimension being the batch size.
        Args:
            imgPts (torch.Tensor): image points, shape (batch_size, num_pts, 2)
            alpha (torch.Tensor): alphas, shape (batch_size, num_pts, 4)
            intrinsics (torch.Tensor): intrinsics, shape (batch_size, 3, 3)
        return
            torch.Tensor: m, shape (batch_size, num_pts * 2, 12)
        """
        batch_size = imgPts.shape[0]
        num_pts = imgPts.shape[1]

        # extract elements of the intrinsic matrix in batch
        fu, fv, u0, v0 = intrinsics[:, 0, 0, None], intrinsics[:, 1, 1, None], intrinsics[:, 0, 2, None], intrinsics[:,
                                                                                                          1, 2, None]
        # extract elements of the image points in batch
        ui, vi = imgPts[:, :, 0], imgPts[:, :, 1]
        # extract elements of the alphas in batch
        a1, a2, a3, a4 = alpha[:, :, 0], alpha[:, :, 1], alpha[:, :, 2], alpha[:, :, 3]
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
        m = m.reshape(batch_size, num_pts * 2, 12)
        return m

    @staticmethod
    def calculate_kernel(m, top=4):
        """Given the m matrix, compute the kernel of it. Check equation 8 in paper for more details.
        Inputs are batched, with first dimension being the batch size.
        Args:
            m (torch.Tensor): m, shape (batch_size, num_pts * 2, 12)
            top (int, optional): number of top eigen vectors to take. Defaults to 4.
        Returns:
            torch.Tensor: kernel, shape (batch_size, 12, top)
        """
        # find null space of M
        eigenvalues, eigenvectors = vmap(torch.linalg.eig)(torch.bmm(m.transpose(1, 2), m))
        # take the real part
        eigenvalues = eigenvalues.real
        eigenvectors = eigenvectors.real
        # sort by eigenvalues (ascending)
        eig_indices = eigenvalues.argsort()
        # take the first 4 eigenvectors, shape (batch_size, 12, 4)
        kernel_bases = torch.gather(eigenvectors, 2, eig_indices[:, :top].unsqueeze(1).tile(1, 12, 1))

        return kernel_bases

    @staticmethod
    def build_l(kernel_bases):
        """Given the kernel of m, compute the L matrix. Check [source](https://github.com/cvlab-epfl/EPnP/blob/5abc3cfa76e8e92e5a8f4be0370bbe7da246065e/cpp/epnp.cpp#L478) for more details.
        Inputs are batched, with first dimension being the batch size.
        Args:
            kernel_bases (torch.Tensor): kernel of m, shape (batch_size, 12, 4)
        Returns:
            torch.Tensor: L, shape (batch_size, 6, 10)
        """
        batch_size = kernel_bases.shape[0]
        kernel_bases = kernel_bases.transpose(1, 2)  # shape (batch_size, 4, 12)
        # calculate the pairwise distance matrix within bases
        diff = kernel_bases.reshape(batch_size, 4, 1, 4, 3) - kernel_bases.reshape(batch_size, 4, 4, 1, 3)
        diff = diff.flatten(start_dim=2, end_dim=3)  # shape (batch_size, 4, 16, 3)
        # six_indices are (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3) before flatten
        dv = diff[:, :, EfficientPnP.six_indices, :]  # shape (batch_size, 4, 6, 3)

        # generate l
        dot_products = torch.sum(
            dv[:, EfficientPnP.ten_indices_pair[0], :, :] * dv[:, EfficientPnP.ten_indices_pair[1], :, :], dim=-1)
        dot_products = dot_products * EfficientPnP.multiply_mask[None, :, None]
        return dot_products.transpose(1, 2)  # shape (batch_size, 6, 10)

    @staticmethod
    def build_rho(contPts_w):
        """Given the coordinates of control points, compute the rho vector. Check [source](https://github.com/cvlab-epfl/EPnP/blob/5abc3cfa76e8e92e5a8f4be0370bbe7da246065e/cpp/epnp.cpp#L520) for more details.
        Inputs are batched, with first dimension being the batch size.
        Args:
            contPts_w (torch.Tensor): coordinates of control points, shape (batch_size, 4, 3)
        Returns:
            torch.Tensor: rho, shape (batch_size, 6, 1)
        """
        dist = contPts_w[:, EfficientPnP.six_indices_pair[0], :] - contPts_w[:, EfficientPnP.six_indices_pair[1], :]
        return torch.sum(dist ** 2, dim=-1)  # l2 norm

    @staticmethod
    def calculate_betas(dim, l, rho):
        """Given the L matrix and rho vector, compute the beta vector. Check equation 10 - 14 in paper for more details.
        Inputs are batched, with first dimension being the batch size.
        Args:
            dim (int): dimension of the problem, 1, 2, or 3
            l (torch.Tensor): L, shape (batch_size, 6, 10)
            rho (torch.Tensor): rho, shape (batch_size, 6)
        """
        if dim == 1:
            betas = torch.zeros(l.shape[0], 4, device=l.device, dtype=l.dtype)
            betas[:, -1] = 1
            return betas
        elif dim == 2:
            l = l[:, :, (5, 8, 9)]  # matched with matlab code
            betas_ = torch.bmm(torch.linalg.pinv(l), rho.unsqueeze(-1)).squeeze(-1)  # shape: (b, 3)
            beta1 = torch.sqrt(torch.abs(betas_[:, 0]))
            beta2 = torch.sqrt(torch.abs(betas_[:, 2])) * torch.sign(betas_[:, 1]) * torch.sign(betas_[:, 0])

            return torch.stack([torch.zeros_like(beta1), torch.zeros_like(beta1), beta1, beta2], dim=-1)
        elif dim == 3:
            l = l[:, :, (2, 4, 7, 5, 8, 9)]  # matched with matlab code
            betas_ = torch.linalg.solve(l, rho.unsqueeze(-1)).squeeze(-1)  # shape: (b, 6)
            beta1 = torch.sqrt(torch.abs(betas_[:, 0]))
            beta2 = torch.sqrt(torch.abs(betas_[:, 3])) * torch.sign(betas_[:, 1]) * torch.sign(betas_[:, 0])
            beta3 = torch.sqrt(torch.abs(betas_[:, 5])) * torch.sign(betas_[:, 2]) * torch.sign(betas_[:, 0])

            return torch.stack([torch.zeros_like(beta1), beta1, beta2, beta3], dim=-1)
        elif dim == 4:
            betas_ = torch.bmm(torch.linalg.pinv(l), rho.unsqueeze(-1)).squeeze(-1)  # shape: (b, 10)
            beta4 = torch.sqrt(abs(betas_[:, 0]))
            beta3 = torch.sqrt(abs(betas_[:, 2])) * torch.sign(betas_[:, 1]) * torch.sign(betas_[:, 0])
            beta2 = torch.sqrt(abs(betas_[:, 5])) * torch.sign(betas_[:, 3]) * torch.sign(betas_[:, 0])
            beta1 = torch.sqrt(abs(betas_[:, 9])) * torch.sign(betas_[:, 6]) * torch.sign(betas_[:, 0])

            return torch.stack([beta1, beta2, beta3, beta4], dim=-1)

    @staticmethod
    def compute_norm_sign_scaling_factor(Xc, alphas, objPts):
        """Compute the scaling factor and the sign of the scaling factor
        Args:
            Xc (torch.tensor): the control points in the camera coordinates, or the result from null space.
            alphas (torch.tensor): the weights of the control points to recover the object points
            objPts (torch.tensor): the object points in the world coordinates
        Returns:
            contPts_c (torch.tensor): the control points in the camera coordinates
            objPts_c (torch.tensor): the object points in the camera coordinates
            sc (torch.tensor): the scaling factor
        """
        # Calculate the control points and object points in the camera coordinates
        contPts_c = Xc.reshape((Xc.shape[0], 4, 3))
        objPts_c = torch.bmm(alphas, contPts_c)

        # Calculate the distance of the reference points in the world coordinates
        objPts_w_centered = objPts - objPts.mean(dim=1, keepdim=True)
        dist_w = torch.linalg.norm(objPts_w_centered, dim=2)

        # Calculate the distance of the reference points in the camera coordinates
        objPts_c_centered = objPts_c - objPts_c.mean(dim=1, keepdim=True)
        dist_c = torch.linalg.norm(objPts_c_centered, dim=2)

        # calculate the scaling factors
        # print(contPts_c)
        sc_1 = torch.bmm(dist_c.unsqueeze(1), dist_c.unsqueeze(2))
        sc_2 = torch.bmm(dist_c.unsqueeze(1), dist_w.unsqueeze(2))
        sc = (1 / sc_1 * sc_2)

        # Update the control points and the object points in the camera coordinates based on the scaling factors
        contPts_c = contPts_c * sc
        objPts_c = torch.matmul(alphas, contPts_c)

        # Update the control points and the object points in the camera coordinates based on the sign
        neg_z_mask = torch.any(objPts_c[:, :, 2] < 0, dim=-1)  # (N, )
        negate_switch = torch.ones((objPts.shape[0],), dtype=objPts.dtype, device=objPts.device)
        negate_switch[neg_z_mask] = negate_switch[neg_z_mask] * -1
        objPts_c = objPts_c * negate_switch.unsqueeze(1).unsqueeze(1)
        sc = sc[:, 0, 0] * negate_switch
        return contPts_c, objPts_c, sc

    @staticmethod
    def get_rotation_translation(objpts_w, objpts_c):
        """
        Get the rotation matrix and translation vector based on the object points in world coordinate and camera coordinate.
        Args:
            objpts_w: The object points in world coordinate. The shape is (B, N, 3).
            objpts_c: The object points in camera coordinate. The shape is (B, N, 3).
        Returns:
            R: The rotation matrix. The shape is (B, 3, 3).
            T: The translation vector. The shape is (B, 3).
        """
        # Get the centered points
        center_w = objpts_w.mean(dim=1, keepdim=True)
        objpts_w = objpts_w - center_w
        center_c = objpts_c.mean(dim=1, keepdim=True)
        objpts_c = objpts_c - center_c

        # Calculate the rotation matrix
        M = vmap(torch.bmm)(objpts_c[:, :, :, None], objpts_w[:, :, None, :])
        M = M.sum(dim=1)  # along the point dimension
        U, S, V = vmap(torch.svd)(M)
        R = torch.bmm(U, V.transpose(dim0=-1, dim1=-2))

        # if det(R) < 0, make it positive
        negate_mask = torch.linalg.det(R) < 0
        R[negate_mask] = -R[negate_mask]

        # Calculate the translation vector based on the rotation matrix and the equation
        T = center_c.transpose(dim0=-1, dim1=-2) - torch.bmm(R, center_w.transpose(dim0=-1, dim1=-2))
        T = T.squeeze(dim=-1)

        return R, T

    @staticmethod
    def reprojection_error(objPts_w, imgPts, camMat, Rt):
        """
        Calculate the reprojection error.
        Args:
            objPts_w: The object points in world coordinate. The shape is (B, N, 3).
            imgPts: The image points. The shape is (B, N, 2).
            camMat: The camera matrix. The shape is (B, 3, 3).
            Rt: The rotation matrix and translation vector. The shape is (B, 3, 4).
        Returns:
            error: The reprojection error. The shape is (B, ).
        """
        P = torch.bmm(camMat[:, :, :3], Rt)
        # concat 1 to the last column of objPts_w
        objPts_w_ex = torch.cat((objPts_w, torch.ones_like(objPts_w[:, :, :1])), dim=-1)
        # Calculate the image points
        imgRep = torch.bmm(P, objPts_w_ex.transpose(dim0=-1, dim1=-2)).transpose(dim0=-1, dim1=-2)

        # Normalize the image points
        imgRep = imgRep[:, :, :2] / imgRep[:, :, 2:]

        error = torch.linalg.norm(imgRep - imgPts, dim=-1)
        error = torch.mean(error, dim=-1)

        return error


class OptimizeBetas(torch.nn.Module):
    def __init__(self, betas):
        super(OptimizeBetas, self).__init__()
        self.betas = torch.nn.Parameter(betas)

    def forward(self, contPts_w, kernel_bases):
        """
        Optimize the betas according to the objectives in the paper.
        For the details, please refer to equation 15.
        Args:
            contPts_w: The control points in world coordinate. The shape is (B, 4, 3).
            kernel_bases: The kernel bases. The shape is (B, 16, 4).
        Returns:
            loss: The loss. The shape is (B, ).
        """
        batch_size = kernel_bases.shape[0]
        # calculate the control points in camera coordinate
        contPts_c = torch.bmm(kernel_bases, self.betas.unsqueeze(-1)).squeeze(-1)
        diff_c = contPts_c.reshape(batch_size, 1, 4, 3) - contPts_c.reshape(batch_size, 4, 1, 3)
        diff_c = diff_c.reshape(batch_size, 16, 3)
        diff_c = torch.sum(diff_c ** 2, dim=-1)

        # calculate the distance between control points in world coordinate
        diff_w = contPts_w.reshape(batch_size, 1, 4, 3) - contPts_w.reshape(batch_size, 4, 1, 3)
        diff_w = diff_w.reshape(batch_size, 16, 3)
        diff_w = torch.sum(diff_w ** 2, dim=-1)

        error = torch.abs(diff_w - diff_c)
        error = torch.mean(error, dim=-1)

        return error
