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

        # get number of dimensions
        m = A.shape[1]

        # translate points to their centroids
        centroid_A = np.mean(A, axis=0)
        centroid_B = np.mean(B, axis=0)
        AA = A - centroid_A
        BB = B - centroid_B

        # rotation matrix
        H = np.dot(AA.T, BB)
        U, S, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)

        # special reflection case
        if np.linalg.det(R) < 0:
           Vt[m-1, :] *= -1
           R = np.dot(Vt.T, U.T)

        # translation
        t = centroid_B.T - np.dot(R, centroid_A.T)

        # homogeneous transformation
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

        # get number of dimensions
        m = A.shape[1]

        # make points homogeneous, copy them to maintain the originals
        src = np.ones((m+1, A.shape[0]))
        dst = np.ones((m+1, B.shape[0]))
        src[:m, :] = np.copy(A.T)
        dst[:m, :] = np.copy(B.T)

        # apply the initial pose estimation
        if init_pose is not None:
            src = np.dot(init_pose, src)

        prev_error = 0

        for i in range(max_iterations):
            # find the nearest neighbors between the current source and destination points
            distances, indices = self.best_fit_transform(src[:m, :].T, dst[:m, :].T)

            # compute the transformation between the current source and nearest destination points
            T, _, _ = self.best_fit_transform(src[:m, :].T, dst[:m, indices].T)

            # update the current source
            src = np.dot(T, src)

            # check error
            mean_error = np.mean(distances)
            if np.abs(prev_error - mean_error) < tolerance:
                break
            prev_error = mean_error

        # calculate final transformation
        T, _, _ = self.best_fit_transform(A, src[:m, :].T)

        return T



