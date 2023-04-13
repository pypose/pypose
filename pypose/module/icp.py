import torch
import pypose as pp
from torch import nn


class ICP(nn.Module):
    r'''
    Iterative Closest Point (ICP) using Singular Value Decomposition (SVD).

    Args:
        steplim: the max step number
        tol: the tolerance of error for early stopping
        matched: whether the input points set and the target points set have been matched
    '''

    def __init__(self, steplim=200, tol=0.0001, matched=False):
        super().__init__()
        self.steplim = steplim
        self.tol = tol
        self.matched = matched

    def forward(self, newpc, originpc):
        temp = newpc.clone()
        iteration = 0
        err = 0
        if (not self.matched):
            while iteration <= self.steplim:
                iteration += 1
                neighbor = self.nearest_neighbor(temp, originpc)
                errnew = neighbor.values.mean()
                transR, transT = self.get_transform(temp, originpc[:, neighbor.indices[-1], :].squeeze(-2))
                temp = temp @ transR + transT
                if (abs(err - errnew) < self.tol):
                    break
                err = errnew

            transR, transT = self.get_transform(newpc, temp)
            transT = torch.transpose(transT, 1, 2)
            T = torch.cat([transR, transT], dim=2)
            return pp.mat2SE3(T, check=False)
        else:
            transR, transT = self.get_transform(newpc, originpc)
            transT = torch.transpose(transT, 1, 2)
            T = torch.cat([transR, transT], dim=2)
            return pp.mat2SE3(T)

    def get_transform(self, p1: torch.tensor, p2: torch.tensor):
        r'''Using SVD algorithm to calculate the transformation matrix between the corresponding points set p1, p2.

        Args:
            p1: input points set
            p2: the matched target points set

        Returns:
            R: the rotation matrix between p1 and p2
            T: the translation matrix between p1 and p2
        '''
        p1_centroid = p1.mean(-2).unsqueeze(-2)
        p2_centroid = p2.mean(-2).unsqueeze(-2)

        temp_p1 = (p1 - p1_centroid)
        temp_p2 = (p2 - p2_centroid)

        H = torch.transpose(temp_p2, 1, 2) @ temp_p1
        u, s, vT = torch.linalg.svd(H)
        v = torch.transpose(vT, 1, 2)
        uT = torch.transpose(u, 1, 2)
        v[:, 2, 2] = v[:, 2, 2] * torch.det(v @ uT)
        vT = torch.transpose(v, 1, 2)
        RT = u @ vT
        R = torch.transpose(RT, 1, 2)
        t = p2_centroid - p1_centroid @ R

        return R, t

    def nearest_neighbor(self, p1, p2, k=1):
        r'''
        Select the nearest neighbor point of p1 from p2
        Args:
            p1: the source points set
            p2: the target points set
            tolerance: the threshold of min distance

        Returns:
            distances: the min distance between point in p1 and its nearest neighbor
            indices: the index of the nearest neighbor point in p2
        '''
        dif = torch.stack([p2[i].unsqueeze(-2) - p1[i]
                           for i in range(p1.shape[0])])

        dist = torch.norm(dif, dim=-1)
        nn = dist.topk(k, largest=False)
        return nn


if __name__=="__main__":
    input_pc = torch.randn([10, 20, 3])
    transT = 0.1 * torch.randn([10, 1, 3])
    print("The ori transT is", transT)
    transR = pp.randn_SO3(10)
    print("The ori transR is", transR)
    transR = transR.matrix()

    output_pc = (input_pc @ transR) + transT

    icpsvd = ICP(matched=False)
    result = icpsvd.forward(input_pc, output_pc)
    print("The ICP result is", result)
