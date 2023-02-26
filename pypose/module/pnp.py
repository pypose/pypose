import torch
import functorch
from functools import partial
from functorch import vmap


class EPnP():
    """
        EPnP Solver - a non-iterative O(n) solution to the PnP problem.
        as described in:

        Francesc Moreno-Noguer, Vincent Lepetit, Pascal Fua.
        Accurate Non-Iterative O(n) Solution to the PnP Problem.
        In Proceedings of ICCV, 2007.
        source: https://github.com/cvlab-epfl/EPnP

    """

    def __init__(self, distCoeff=None):
        '''
        Args:
            objPts: Vectors fo the reference points in the world coordinate.
            imgPts: Vectors of the projection of the reference points. 
            camMat: Camera intrinsic matrix.
            distCoeff: Distortion matrix.

        Returns:
            Rt: Transform matrix include the rotation and the translation [R|t].
        '''
        # TODO: Ensure objPts / imgPts take in batched inputs
        self.disCoeff = distCoeff

    def forward(self, objPts, imgPts, intrinsics):
        # Select four control points and calculate alpha
        contPts_w = self.select_control_points(objPts)  # Select 4 control points (in the world coordinate)
        alpha = self.compute_alphas(objPts, contPts_w)

        # Using camera projection equation for all the points pairs to get the matrix M
        m = self.build_m(imgPts, alpha, intrinsics)

        kernel_m = self.calculate_kernel(m)
        # dim(kernel_space) = 1
        self.compute_norm_sign_scaling_factor(kernel_m[:, :, 0], contPts_w, alpha, objPts)

        return
    def main_EPnP(self):

        fu, fv, u0, v0 = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[
            1, 2]  # Elements of the intrinsic matrix

        # Calculate the eigenvectors of (M^T)M
        M_T_M = np.matmul(M.transpose(), M)
        M_T_M = np.array(M_T_M, dtype=float)
        W, V = np.linalg.eig(M_T_M)
        idx = W.argsort()
        self.v_mat = V[:, idx[:4]]  # Pick up the four eigen vectors with the smallest four eigen values

        errors = []
        Rt_sol, contPts_c_sol, objPts_c_sol, sc_sol, beta_sol = [], [], [], [], []

        # Form the L matrix for calculating beta, which used to denote the control points in the camera coordinate
        v_mat = self.v_mat
        V_M = np.array([v_mat.T[3], v_mat.T[2], v_mat.T[1], v_mat.T[0]]).T
        L6_10_mat = self.compute_L6_10_mat_mat(V_M)

        # Calculate betas; get the control points in the camrea coordinate; get the rotation and translation
        for i in range(3):
            error, Rt, contPts_c, objPts_c, sc, beta = self.diffDim_calculation(i + 1, v_mat[:, :(i + 1)], L6_10_mat)
            errors.append(error)
            Rt_sol.append(Rt)
            contPts_c_sol.append(contPts_c)
            objPts_c_sol.append(objPts_c)
            sc_sol.append(sc)
            beta_sol.append(beta)

        best = np.array(errors).argsort()[0]
        error_best = errors[best]
        Rt_best, contPts_c_best, objPts_c_best = Rt_sol[best], contPts_c_sol[best], objPts_c_sol[best]
        sc_best, beta_best = sc_sol[best], beta_sol[best]

        # TODO: Separate Gauss Newton to a new class
        # apply gauss-newton optimization
        best = len(beta_best)
        if best == 1:
            Betas = [0, 0, 0, beta_best[0]]
        elif best == 2:
            Betas = [0, 0, beta_best[0], beta_best[1]]
        else:
            Betas = [0, beta_best[0], beta_best[1], beta_best[2]]

        Beta0 = sc_best * np.array(Betas)
        v_mat = self.v_mat
        V_M = np.array([v_mat.T[3], v_mat.T[2], v_mat.T[1], v_mat.T[0]]).T

        objPts_c_opt, contPts_c_opt, Rt_opt, err_opt = self.optimize_betas_gauss_newton(V_M, Beta0)

        if err_opt < error_best:
            error_best, Rt_best, contPts_c_best, objPts_c_best = err_opt, Rt_opt, contPts_c_opt, objPts_c_opt

        return error_best, Rt_best, contPts_c_best, objPts_c_best

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
        res = []
        res.append(center)
        for i in range(3):
            another_pt = center + torch.sqrt(s[:, i, None]) * vh[:, i]
            res.append(another_pt)

        return torch.stack(res, dim=1)

    @staticmethod
    def compute_alphas(objPts, contPts_w, linear_least_square=False):
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

        if linear_least_square:
            NotImplementedError("Linear least square method is not implemented yet.")
            # Calculate Alpha TODO: CHECK if logic is correct, or change to general method
            alpha = torch.bmm(torch.linalg.inv(contPts_w), objPts)  # simple method
            alpha = alpha.transpose()
        else:
            alpha = torch.linalg.solve(contPts_w, objPts, left=False)  # General method

        return alpha

    def build_m(self, imgPts, alpha, intrinsics):
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
        fu, fv, u0, v0 = intrinsics[:, 0, 0, None], intrinsics[:, 1, 1, None], intrinsics[:, 0, 2, None], intrinsics[:, 1, 2, None]
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

    def compute_L6_10_mat_mat(self, V_M):
        L = np.zeros((6, 10))

        # Rearrange the eigen vectors of (M^T)M
        v = []
        for i in range(4):
            v.append(V_M[:, i])

        # Generate a Matrix include all S = v_i - v_j 
        dv = []
        for r in range(4):
            dv.append([])
            for i in range(3):
                for j in range(i + 1, 4):
                    dv[r].append(v[r][3 * i:3 * (i + 1)] - v[r][3 * j:3 * (j + 1)])

        # Generate the L6_10 Matrix
        index = [(0, 0),
                 (0, 1), (1, 1),
                 (0, 2), (1, 2), (2, 2),
                 (0, 3), (1, 3), (2, 3), (3, 3)]
        for i in range(6):
            j = 0
            for a, b in index:
                L[i, j] = np.matmul(dv[a][i], dv[b][i].T)
                if a != b:
                    L[i, j] *= 2
                j += 1

        return L

    def diffDim_calculation(self, N, v_mat, L6_10_mat):
        # Calculate rho - the right hand side of the equation (consist of ||c_i^w - c_j^w||^2: c is the control point in the world coordinate)
        rho = []
        for i in range(3):
            for j in range(i + 1, 4):
                rho.append(np.sum((self.contPts_w[i, :] - self.contPts_w[j, :]) ** 2, axis=0))

        # Calculate beta in different N cases
        if N == 1:
            X1 = v_mat
            contPts_c, Xc, sc = self.compute_norm_sign_scaling_factor(X1)
            beta = [1]

        if N == 2:
            L = L6_10_mat[:, (5, 8, 9)]
            # For Ax = b problem, we use x = inv(A^TA)(A^Tb). 
            # SVD is another (better) approach to solve this Linear least squares problems
            # betas = [beta22, beta12, beta11]
            betas = np.matmul(np.linalg.inv(np.matmul(L.T, L)), np.matmul(L.T, rho))
            beta2 = math.sqrt(abs(betas[0]))
            beta1 = math.sqrt(abs(betas[2])) * np.sign(betas[1]) * np.sign(betas[0])

            X2 = beta2 * v_mat.T[1] + beta1 * v_mat.T[0]
            contPts_c, Xc, sc = self.compute_norm_sign_scaling_factor(X2)
            beta = [beta2, beta1]

        if N == 3:
            L = L6_10_mat[:, (2, 4, 7, 5, 8, 9)]
            # Since the size of L6 matrix is 6*6 when N = 3, we just simply use x = inv(A)b to do calculation
            # betas = [beta33, beta23, beta13, beta22, beta12, beta11]
            betas = np.matmul(np.linalg.inv(L), rho)
            beta3 = math.sqrt(abs(betas[0]))
            beta2 = math.sqrt(abs(betas[3])) * np.sign(betas[1]) * np.sign(betas[0])
            beta1 = math.sqrt(abs(betas[5])) * np.sign(betas[2]) * np.sign(betas[0])

            X3 = beta3 * v_mat.T[2] + beta2 * v_mat.T[1] + beta1 * v_mat.T[0]
            contPts_c, Xc, sc = self.compute_norm_sign_scaling_factor(X3)
            beta = [beta3, beta2, beta1]

        if N == 4:
            L = L6_10_mat
            # For Ax = b problem, we use x = inv(A^TA)(A^Tb). 
            # SVD is another (better) approach to solve this Linear least squares problems
            # betas = [beta44, beta34, beta33, beta24, beta23, beta22, beta14, beta13, beta12, beta11]
            betas = np.matmul(np.linalg.inv(np.matmul(L.T, L)), np.matmul(L.T, rho))
            beta4 = math.sqrt(abs(betas[0]))
            beta3 = math.sqrt(abs(betas[2])) * np.sign(betas[1]) * np.sign(betas[0])
            beta2 = math.sqrt(abs(betas[5])) * np.sign(betas[3]) * np.sign(betas[0])
            beta1 = math.sqrt(abs(betas[9])) * np.sign(betas[6]) * np.sign(betas[0])

            X4 = beta4 * v_mat.T[3] + beta3 * v_mat.T[2] + beta2 * v_mat.T[1] + beta1 * v_mat.T[0]
            contPts_c, Xc, sc = self.compute_norm_sign_scaling_factor(X4)
            beta = [beta4, beta3, beta2, beta1]

        # Get the rotation and the translation
        R, T = self.ICP_getRotT(self.objPts, Xc)
        Rt = np.concatenate((R.reshape((3, 3)), T.reshape((3, 1))), axis=1)
        error = self.reprojection_error(self.objPts, self.imgPts, Rt)

        return error, Rt, contPts_c, Xc, sc, beta

    def compute_norm_sign_scaling_factor(self, Xc, contPts_w, alphas, objPts):
        """Compute the scaling factor and the sign of the scaling factor
        Args:
            Xc (torch.tensor): the control points in the camera coordinates
            contPts_w (torch.tensor): the control points in the world coordinates
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
        sc = 1 / sc_1 * sc_2

        # Update the control points and the object points in the camera coordinates based on the scaling factors
        contPts_c = contPts_c * sc
        objPts_c = torch.matmul(alphas, contPts_c)

        # Update the control points and the object points in the camera coordinates based on the sign
        neg_z_mask = torch.any(objPts_c[:, :, 2] < 0, dim=-1)  # (N, )
        negate_switch = torch.ones((objPts.shape[0], ), dtype=objPts.dtype, device=objPts.device)
        negate_switch[neg_z_mask] = negate_switch[neg_z_mask] * -1
        objPts_c = objPts_c * negate_switch.unsqueeze(1).unsqueeze(1)
        sc = sc * negate_switch
        return contPts_c, objPts_c, sc

    # region "Gauss-Newton OPtimization Block"
    def optimize_betas_gauss_newton(self, V_M, Beta0):
        n = len(Beta0)
        Beta_opt, _ = self.gauss_newton(V_M, Beta0)
        X = np.zeros((12))
        for i in range(n):
            X = X + Beta_opt[i] * V_M[:, i]

        contPts_c = []
        for i in range(4):
            contPts_c.append(X[(3 * i): (3 * (i + 1))])

        contPts_c = np.array(contPts_c).reshape((4, 3))
        s_contPts_w = self.sign_determinant(self.contPts_w)
        s_contPts_c = self.sign_determinant(contPts_c)
        contPts_c = contPts_c * s_contPts_w * s_contPts_c

        Xc_opt = np.matmul(self.Alpha, contPts_c)
        R_opt, T_opt = self.ICP_getRotT(self.objPts, Xc_opt)
        Rt_opt = np.concatenate((R_opt.reshape((3, 3)), T_opt.reshape((3, 1))), axis=1)
        err_opt = self.reprojection_error(self.objPts, self.imgPts, Rt_opt)

        return Xc_opt, contPts_c, Rt_opt, err_opt

    def gauss_newton(self, V_M, Beta0):
        L = self.compute_L6_10_mat_mat(V_M)
        rho = []
        for i in range(3):
            for j in range(i + 1, 4):
                rho.append(np.sum((self.contPts_w[i, :] - self.contPts_w[j, :]) ** 2, axis=0))

        current_betas = Beta0

        # Iteration number of optimization
        n_iterations = 10
        for k in range(n_iterations):
            A, b = self.compute_A_and_b_Gauss_Newton(current_betas, rho, L)
            dbeta = np.matmul(np.linalg.inv(np.matmul(A.T, A)), np.matmul(A.T, b))
            current_betas = current_betas + dbeta.T[0]
            error = np.matmul(b.T, b)

        Beta_opt = current_betas

        return Beta_opt, error

    def compute_A_and_b_Gauss_Newton(self, cb, rho, L):
        A = np.zeros((6, 4))
        b = np.zeros((6, 1))

        B = [cb[0] * cb[0],
             cb[0] * cb[1],
             cb[1] * cb[1],
             cb[0] * cb[2],
             cb[1] * cb[2],
             cb[2] * cb[2],
             cb[0] * cb[3],
             cb[1] * cb[3],
             cb[2] * cb[3],
             cb[3] * cb[3]]

        for i in range(6):
            A[i, 0] = 2 * cb[0] * L[i, 0] + cb[1] * L[i, 1] + cb[2] * L[i, 3] + cb[3] * L[i, 6]
            A[i, 1] = cb[0] * L[i, 1] + 2 * cb[1] * L[i, 2] + cb[2] * L[i, 4] + cb[3] * L[i, 7]
            A[i, 2] = cb[0] * L[i, 2] + cb[1] * L[i, 4] + 2 * cb[2] * L[i, 5] + cb[3] * L[i, 8]
            A[i, 3] = cb[0] * L[i, 3] + cb[1] * L[i, 7] + cb[2] * L[i, 8] + 2 * cb[3] * L[i, 9]

            b[i] = rho[i] - np.matmul(L[i, :], B)

        return A, b

    def sign_determinant(self, C):
        M = []
        for i in range(3):
            M.append(C[i, :].T - C[-1, :].T)

        return np.sign(np.linalg.det(M))

    # endregion

    def ICP_getRotT(self, objpts_w, objpts_c):
        # Find the center points for the reference points in the world coordinates and the camera coordinates respectively
        objpts_w_cent = np.tile(np.mean(objpts_w, axis=0).reshape((1, 3)), (self.n, 1))
        objPts_c_cent = np.tile(np.mean(objpts_c, axis=0).reshape((1, 3)), (self.n, 1))

        # Get the centered points by minus the center points
        objpts_w = objpts_w.reshape((self.n, 3)) - objpts_w_cent
        objpts_c = objpts_c.reshape((self.n, 3)) - objPts_c_cent

        # Calculate the rotation matrix
        M = np.matmul(objpts_c.transpose(), objpts_w)
        U, S, V = np.linalg.svd(M)
        R = np.matmul(U, V)

        # When the result matrix's determinate value is negative then make it positive to make sure R is a rotation matrix
        if np.linalg.det(R) < 0:
            R = - R

        # Calculate the translation vector based on the rotation matrix and the equation
        T = objPts_c_cent[0].transpose() - np.matmul(R, objpts_w_cent[0].transpose())

        return R, T

    def reprojection_error(self, objPts_w, imgPts, Rt):
        P = np.matmul(self.camMat[:, :3], Rt)
        objPts_w_ex = np.concatenate((objPts_w.reshape((self.n, 3)), np.array([np.ones((self.n))]).T), axis=1)

        imgRep = np.matmul(P, objPts_w_ex.T).T
        imgRep[:, 0] = imgRep[:, 0] / imgRep[:, 2]
        imgRep[:, 1] = imgRep[:, 1] / imgRep[:, 2]
        error = np.sqrt((imgPts[:, 0] - imgRep[:, 0].reshape((self.n, 1))) ** 2 + (
                    imgPts[:, 1] - imgRep[:, 1].reshape((self.n, 1))) ** 2)
        error = np.sum(error, axis=0) / self.n

        return error[0]
