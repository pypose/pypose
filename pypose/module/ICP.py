import torch
import pypose as pp
from torch import nn
import numpy as np
from sklearn.neighbors import NearestNeighbors

class ICP(nn.Module):
    def __init__(self, A, B):
        super().__init__()
        self.get_trans = self.icp(A, B)

    def best_fit_transform(self, A, B):
        assert A.shape == B.shape
        m = A.shape[1]
        centroid_A = np.mean(A, axis=0)
        centroid_B = np.mean(B, axis=0)
        AA = A - centroid_A
        BB = B - centroid_B
        H = np.dot(AA.T, BB)
        U, S, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)
        if np.linalg.det(R) < 0:
           Vt[m-1, :] *= -1
           R = np.dot(Vt.T, U.T)
        t = centroid_B.T - np.dot(R, centroid_A.T)
        T = np.identity(m+1)
        T[:m, :m] = R
        T[:m, m] = t

        return T, R, t

    def nearest_neighbor(self, src, dst):
        assert src.shape == dst.shape
        neigh = NearestNeighbors(n_neighbors=1)
        neigh.fit(dst)
        distances, indices = neigh.kneighbors(src, return_distance=True)

        return distances.ravel(), indices.ravel()

    def icp(self, A, B, init_pose=None, max_iterations=20, tolerance=0.001):

        assert A.shape == B.shape
        m = A.shape[1]
        src = np.ones((m+1, A.shape[0]))
        dst = np.ones((m+1, B.shape[0]))
        src[:m, :] = np.copy(A.T)
        dst[:m, :] = np.copy(B.T)
        if init_pose is not None:
            src = np.dot(init_pose, src)
        prev_error = 0
        for i in range(max_iterations):
            distances, indices = self.best_fit_transform(src[:m, :].T, dst[:m, :].T)
            T, _, _ = self.best_fit_transform(src[:m, :].T, dst[:m, indices].T)
            src = np.dot(T, src)
            mean_error = np.mean(distances)
            if np.abs(prev_error - mean_error) < tolerance:
                break
            prev_error = mean_error
        T, _, _ = self.best_fit_transform(A, src[:m, :].T)

        return T



