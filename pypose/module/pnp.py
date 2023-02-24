import torch

class EPnP():
    '''
    EPnP Solver - a non-iterative O(n) solution to the PnP problem.

    Author:
        Yi Du
    '''

    def __init__(self, objPts=None, imgPts=None, camMat=None, distCoeff=None):
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
        self.objPts = objPts.reshape((objPts.shape[0], 3, 1))
        self.imgPts = imgPts.reshape((objPts.shape[0], 2, 1))
        self.camMat = camMat
        self.disCoeff = distCoeff
        self.n = len(self.objPts) # Number of points

    def forward(self):
        # TODO: Modify this to handle obj pts and img pts inputs
        raise NotImplementedError


    def main_EPnP(self):
        # Select four control points and calculate alpha
        self.contPts_w = self.select_control_points() # Select 4 control points (in the world coordinate)
        self.Alpha = self.compute_alphas()

        # Using camera projection equation for all the points pairs to get the matrix M  
        A, Alpha = self.camMat, self.Alpha
        fu, fv, u0, v0 = A[0, 0], A[1, 1], A[0, 2], A[1, 2] # Elements of the intrinsic matrix
        M = []
        imgPts = np.array(self.imgPts)
        for i in range(self.n):
            M.append([Alpha[i, 0] * fu, 0, Alpha[i, 0] * (u0 - imgPts[i, 0]), 
                      Alpha[i, 1] * fu, 0, Alpha[i, 1] * (u0 - imgPts[i, 0]),
                      Alpha[i, 2] * fu, 0, Alpha[i, 2] * (u0 - imgPts[i, 0]),
                      Alpha[i, 3] * fu, 0, Alpha[i, 3] * (u0 - imgPts[i, 0])])
            M.append([0, Alpha[i, 0] * fv, Alpha[i, 0] * (v0 - imgPts[i, 1]), 
                      0, Alpha[i, 1] * fv, Alpha[i, 1] * (v0 - imgPts[i, 1]),
                      0, Alpha[i, 2] * fv, Alpha[i, 2] * (v0 - imgPts[i, 1]),
                      0, Alpha[i, 3] * fv, Alpha[i, 3] * (v0 - imgPts[i, 1])])
        M = np.array(M)

        # Calculate the eigenvectors of (M^T)M
        M_T_M = np.matmul(M.transpose(), M)
        M_T_M = np.array(M_T_M, dtype=float)
        W, V = np.linalg.eig(M_T_M)
        idx = W.argsort()
        self.v_mat = V[:, idx[:4]] # Pick up the four eigen vectors with the smallest four eigen values
        
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
        
    def select_control_points(self):
        # Select the center of the mass to be the first control point
        contPts_w_1 = np.mean(self.objPts, axis=0).reshape((1, 3))
        center_objPts = np.tile(contPts_w_1, (self.n, 1))
        
        # Use the first control point and PCA to select the other three control points
        objPts_w_cent = self.objPts.reshape((self.n, 3)) - center_objPts
        u, s, vh = np.linalg.svd(np.matmul(objPts_w_cent.T, objPts_w_cent), full_matrices=True)
        contPts_w_2 = contPts_w_1 + np.sqrt(s[0])*vh[0]
        contPts_w_3 = contPts_w_1 + np.sqrt(s[1])*vh[1]
        contPts_w_4 = contPts_w_1 + np.sqrt(s[2])*vh[2]

        return np.array([contPts_w_1, contPts_w_2, contPts_w_3, contPts_w_4]).reshape(4, 3)
        
    def compute_alphas(self):
        # Construct matrix for alpha calculation
        objPts_w = np.array(self.objPts).transpose()[0]
        mat_objPts_w = np.concatenate((objPts_w, np.array([np.ones((self.n))])), axis=0)
        contPts_w = self.contPts_w.transpose()
        mat_contPts_w = np.concatenate((contPts_w, np.array([np.ones((4))])), axis=0)
        
        # Calculate Alpha
        Alpha = np.matmul(np.linalg.inv(mat_contPts_w), mat_objPts_w) # simple method
        Alpha = Alpha.transpose()
        # Alpha = solve(mat_contPts_w, mat_objPts_w) # General method
        # Alpha = Alpha.transpose()
        
        return Alpha
    
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
                for j in range(i+1, 4):
                    dv[r].append(v[r][3*i:3*(i+1)] - v[r][3*j:3*(j+1)])

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
            
    def compute_norm_sign_scaling_factor(self, Xc):
        contPts_c = []
    
        for i in range(4):
            contPts_c.append(Xc[(3 * i) : (3 * (i + 1))])
        
        # Calculate the control points and object points in the camera coordinates
        contPts_c = np.array(contPts_c).reshape((4, 3))
        objPts_c = np.matmul(self.Alpha, contPts_c)
        
        # Calculate the distance of the reference points in the world coordinates
        centr_w = np.mean(self.objPts, axis=0)
        centroid_w = np.tile(centr_w.reshape((1, 3)), (self.n, 1))
        objPts_w_centered = self.objPts.reshape((self.n, 3)) - centroid_w
        dist_w = np.sqrt(np.sum(objPts_w_centered ** 2, axis=1))
        
        # Calculate the distance of the reference points in the camera coordinates
        centr_c = np.mean(np.array(objPts_c), axis=0)
        centroid_c = np.tile(centr_c.reshape((1, 3)), (self.n, 1))
        objPts_c_centered = objPts_c.reshape((self.n, 3)) - centroid_c
        dist_c = np.sqrt(np.sum(objPts_c_centered ** 2, axis=1))
        
        # calculate the scaling factors
        # print(contPts_c)
        sc_1 = np.matmul(dist_c.transpose(), dist_c) ** -1
        sc_2 = np.matmul(dist_c.transpose(), dist_w)
        sc = sc_1 * sc_2
        
        # Update the control points and the object points in the camera coordinates based on the scaling factors
        contPts_c *= sc
        objPts_c = np.matmul(self.Alpha, contPts_c)
    
        # Update the control points and the object points in the camera coordinates based on the sign    
        for x in objPts_c:
            if x[-1] < 0:
                objPts_c *= -1
                contPts_c *= -1
        
        return contPts_c, objPts_c, sc
    
    #region "Gauss-Newton OPtimization Block"
    def optimize_betas_gauss_newton(self, V_M, Beta0):
        n = len(Beta0)
        Beta_opt, _ = self.gauss_newton(V_M, Beta0)
        X = np.zeros((12))
        for i in range(n):
            X = X + Beta_opt[i] * V_M[:, i]
        
        contPts_c = []
        for i in range(4):
            contPts_c.append(X[(3 * i) : (3 * (i + 1))])
        
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
        
        B=[cb[0] * cb[0],
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
    #endregion

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
        error = np.sqrt((imgPts[:, 0] - imgRep[:, 0].reshape((self.n, 1))) ** 2 + (imgPts[:, 1] - imgRep[:, 1].reshape((self.n, 1))) ** 2)
        error = np.sum(error, axis=0) / self.n

        return error[0]

   