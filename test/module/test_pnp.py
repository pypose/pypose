import os
import pypose as pp
import torch
import urllib
import numpy as np

import logging

logger = logging.getLogger(__name__)


def fetch_epfl_example():
    # load epfl's mat file
    test_mat_url = 'https://github.com/cvlab-epfl/EPnP/raw/master/matlab/data/input_data_noise.mat'
    tmp_path = '/tmp/pypose_test/input_data_noise.mat'
    os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
    if not os.path.exists(tmp_path):
        urllib.request.urlretrieve(test_mat_url, tmp_path)
    assert os.path.exists(tmp_path), 'download testing mat failed'
    return tmp_path


def load_data():
    keys = ['camPts', 'Ximg_true', 'imgPts_true', 'Ximg', 'imgPts', 'objPts']
    res = torch.load('epfl_sample.pth')

    for key in keys:
        res[key] = res[key].squeeze(-1)
        # make a copy of the data for batch size of 2
        res[key] = res[key].unsqueeze(0)[[0, 0, ]]

    res['camMat'] = res['camMat'].unsqueeze(0)[[0, 0, ]].to(res['objPts'])
    res['Rt'] = res['Rt'].unsqueeze(0)[[0, 0, ]]
    return res


def rmse_rot(pred, gt):
    diff = pred - gt
    f_norm = torch.norm(diff, dim=(1, 2))
    return f_norm.mean()


def rmse_t(pred, gt):
    diff = pred - gt
    norm = diff ** 2
    norm = torch.sum(norm, dim=1)
    return norm.mean()


class TestEPnP:
    def test_epnp_5pts(self):
        # TODO: Implement this
        return

    def test_epnp_10pts(self):
        # TODO: Implement this
        return

    def test_epnp_random(self):
        def solution_opencv(obj_pts, img_pts, intrinsics):
            distortion = None
            # results given by cv2.solvePnP(obj_pts, img_pts, intrinsics, distortion, flags=cv2.SOLVEPNP_EPNP)
            rvec = np.array([[-0.37279579],
                             [0.09247041],
                             [-0.82372009]])
            t = np.array([[-0.07126862],
                          [-0.22845308],
                          [6.92614964]])
            rot = np.array([[0.67947332, 0.69882627, 0.22351252],
                            [-0.73099025, 0.61862761, 0.28801584],
                            [0.06300202, -0.35908455, 0.93117615]])
            Rt = np.concatenate((rot.reshape((3, 3)), t.reshape((3, 1))), axis=1)

            obj_pts = torch.from_numpy(obj_pts[None])
            img_pts = torch.from_numpy(img_pts[None])
            intrinsics = torch.from_numpy(intrinsics[None])
            Rt = torch.from_numpy(Rt[None])

            rot = Rt[:, :3, :3][None]
            t = Rt[:, :3, 3][None]

            error = pp.module.EPnP.reprojection_error(obj_pts,
                                                      img_pts,
                                                      intrinsics,
                                                      Rt, )
            return dict(Rt=Rt, error=error, R=rot, T=t)

        data = load_data()

        # instantiate epnp
        epnp = pp.module.EPnP(refinement_optimizer=False)
        solution = epnp.forward(data['objPts'], data['imgPts'], data['camMat'])
        solution_ref = solution_opencv(data['objPts'][0].numpy(),
                                       data['imgPts'][0].numpy(),
                                       data['camMat'][0].numpy())
        gt_rot = data['Rt'][:, :3, :3]
        gt_t = data['Rt'][:, :3, 3]

        print("Pypose EPnP solution, rmse of R:", rmse_rot(solution['R'], gt_rot))
        print("Pypose EPnP solution, rmse of t:", rmse_t(solution['t'], gt_t))

        print("OpenCV EPnP solution, rmse of R:", rmse_rot(solution_ref['R'], gt_rot))
        print("OpenCV EPnP solution, rmse of t:", rmse_t(solution_ref['T'], gt_t))


# CPnP: a consistent PnP solver
# Inputs: s - a 3×n matrix whose i-th column is the coordinates (in the world frame) of the i-th 3D point
# Psens_2D - a 2×n matrix whose i-th column is the coordinates of the 2D projection of the i-th 3D point
# fx, fy, u0, v0 - intrinsics of the camera, corresponding to the intrinsic matrix K=[fx 0 u0;0 fy v0;0 0 1]

# Outputs: R - the estimate of the rotation matrix in the first step
# t - the estimate of the translation vector in the first step
# R_GN - the refined estimate of the rotation matrix with Gauss-Newton iterations
# t_GN - the refined estimate of the translation vector with Gauss-Newton iterations
# Copyright <2022> <Guangyang Zeng, Shiyu Chen, Biqiang Mu, Guodong Shi, Junfeng Wu>
# Guangyang Zeng, SLAMLab-CUHKSZ, September 2022
# zengguangyang@cuhk.edu.cn, https://github.com/SLAMLab-CUHKSZ
# paper link: https://arxiv.org/abs/2209.05824


from numpy import linalg
# from scipy.linalg import expm, eigh, eig, svd

#
# def CPnP(s, Psens_2D, fx, fy, u0, v0):
#     """
#     This is the official implementation
#     """
#     N = s.shape[1]
#     bar_s = np.mean(s, axis=1).reshape(3, 1)
#     Psens_2D = Psens_2D - np.array([[u0], [v0]])
#     obs = Psens_2D.reshape((-1, 1), order="F")
#     pesi = np.zeros((2 * N, 11))
#     G = np.ones((2 * N, 1))
#     W = np.diag([fx, fy])
#     M = np.hstack([np.kron(bar_s.T, np.ones((2 * N, 1))) - np.kron(s.T, np.ones((2, 1))), np.zeros((2 * N, 8))])
#
#     for k in range(N):
#         pesi[[2 * k], :] = np.hstack(
#             [-(s[0, k] - bar_s[0]) * obs[2 * k], -(s[1, k] - bar_s[1]) * obs[2 * k], -(s[2, k] - bar_s[2]) * obs[2 * k],
#              (fx * s[:, [k]]).T.tolist()[0], fx, 0, 0, 0, 0])
#         pesi[[2 * k + 1], :] = np.hstack(
#             [-(s[0, k] - bar_s[0]) * obs[2 * k + 1], -(s[1, k] - bar_s[1]) * obs[2 * k + 1],
#              -(s[2, k] - bar_s[2]) * obs[2 * k + 1], 0, 0, 0, 0, (fy * s[:, [k]]).T.tolist()[0], fy])
#
#     J = np.dot(np.vstack([pesi.T, obs.T]), np.hstack([pesi, obs])) / (2 * N)
#     delta = np.vstack([np.hstack([np.dot(M.T, M), np.dot(M.T, G)]), np.hstack([np.dot(G.T, M), np.dot(G.T, G)])]) / (
#             2 * N)
#
#     w, D = eig(J, delta)
#     sigma_est = min(abs(w))
#
#     est_bias_eli = np.dot(np.linalg.inv((np.dot(pesi.T, pesi) - sigma_est * (np.dot(M.T, M))) / (2 * N)),
#                           (np.dot(pesi.T, obs) - sigma_est * np.dot(M.T, G)) / (2 * N))
#     bias_eli_rotation = np.vstack([est_bias_eli[3:6].T, est_bias_eli[7:10].T, est_bias_eli[0:3].T])
#     bias_eli_t = np.hstack([est_bias_eli[6], est_bias_eli[10],
#                             1 - bar_s[0] * est_bias_eli[0] - bar_s[1] * est_bias_eli[1] - bar_s[2] * est_bias_eli[2]]).T
#     normalize_factor = np.linalg.det(bias_eli_rotation) ** (1 / 3)
#     bias_eli_rotation = bias_eli_rotation / normalize_factor
#     t = bias_eli_t / normalize_factor
#
#     U, x, V = svd(bias_eli_rotation)
#     V = V.T
#
#     RR = np.dot(U, np.diag([1, 1, np.linalg.det(np.dot(U, V.T))]))
#     R = np.dot(RR, V.T)
#
#     E = np.array([[1, 0, 0], [0, 1, 0]])
#     WE = np.dot(W, E)
#     e3 = np.array([[0], [0], [1]])
#     J = np.zeros((2 * N, 6))
#
#     g = np.dot(WE, np.dot(R, s) + np.tile(t, N).reshape(N, 3).T)
#     h = np.dot(e3.T, np.dot(R, s) + np.tile(t, N).reshape(N, 3).T)
#
#     f = g / h
#     f = f.reshape((-1, 1), order="F")
#     I3 = np.diag([1, 1, 1])
#
#     for k in range(N):
#         J[[2 * k, 2 * k + 1], :] = np.dot((WE * h[0, k] - g[:, [k]] * e3.T), np.hstack(
#             [s[1, k] * R[:, [2]] - s[2, k] * R[:, [1]], s[2, k] * R[:, [0]] - s[0, k] * R[:, [2]],
#              s[0, k] * R[:, [1]] - s[1, k] * R[:, [0]], I3])) / h[0, k] ** 2
#
#     initial = np.hstack([np.zeros((3)), t.tolist()]).reshape(6, 1)
#     results = initial + np.dot(np.dot(np.linalg.inv(np.dot(J.T, J)), J.T), (obs - f))
#     X_GN = results[0:3]
#     t_GN = results[3:6]
#
#     X_GN = X_GN.reshape(3, )
#
#     Xhat = np.array([
#         [0, -X_GN[2], X_GN[1]],
#         [X_GN[2], 0, -X_GN[0]],
#         [-X_GN[1], X_GN[0], 0]
#     ])
#     R_GN = np.dot(R, expm(Xhat))
#
#     return R, t, R_GN, t_GN
#
#
# class TestCPnP:
#     def cpnp_wrapper(self, data):
#         obj_pts = data["objPts"][0].numpy().T
#         img_pts = data["imgPts"][0].numpy().T
#         intrinsic = data["camMat"][0].numpy().T
#
#         fx, fy, u0, v0 = intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]
#         solution_ = CPnP(obj_pts, img_pts, fx, fy, u0, v0)
#         keys = ['R', 't', 'R_GN', 't_GN', ]
#         solution = {key: solution_[i] for i, key in enumerate(keys)}
#         # to torch
#         for key in solution.keys():
#             solution[key] = torch.from_numpy(solution[key])[None, :]
#         return solution
#
#     def test_cpnp_random(self):
#         data = load_data()
#         solution = self.cpnp_wrapper(data)
#         R = solution['R']
#         t = solution['t']
#         R_GN = solution['R_GN']
#         t_GN = solution['t_GN']
#         gt_rot = data['Rt'][:, :3, :3]
#         gt_t = data['Rt'][:, :3, 3]
#
#         print("Official CPnP solution, rmse of R:", rmse_rot(solution['R'], gt_rot))
#         print("Official CPnP solution, rmse of t:", rmse_t(solution['t'], gt_t))
#
#         print("Official CPnP solution w/ GN, rmse of R:", rmse_rot(solution['R_GN'], gt_rot))
#         print("Official CPnP solution w/ GN, rmse of t:", rmse_t(solution['t_GN'].squeeze(-1), gt_t))


if __name__ == "__main__":
    epnp_fixture = TestEPnP()
    epnp_fixture.test_epnp_5pts()
    epnp_fixture.test_epnp_10pts()
    epnp_fixture.test_epnp_random()

    # cpnp_fixture = TestCPnP()
    # cpnp_fixture.test_cpnp_random()
