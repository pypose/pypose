import torch
import pypose as pp
from torch import nn


class ICP(nn.Module):
    r'''
    Iterative Closest Point (ICP) using Singular Value Decomposition (SVD).

    Args:
        max_iterations: the max iteration number
        tolerance: the tolerance of error for early stopping
        is_matching: whether the input points set and the target points set have been matched
    '''

    def __init__(self, max_iterations=200, tolerance=0.0001, is_matching=False):
        super().__init__()
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.is_matching = is_matching

    def forward(self, p1, p2):
        temp_pc = p1.clone()
        iter_num = 0
        prev_err = 0
        if (not self.is_matching):
            while iter_num <= self.max_iterations:
                iter_num += 1
                distances, indices = self.nearest_neighbor(temp_pc, p2)
                mean_err = sum(distances) / len(distances)
                transR, transT = self.get_transform(temp_pc, p2[indices])
                temp_pc = torch.matmul(temp_pc, transpose(transR)) + transT
                if (abs(prev_err - mean_err) < self.tolerance):
                    break
                prev_err = mean_err
            transR, transT = self.get_transform(p1, temp_pc)
            return self.RT2SE3(transR, transT)
        else:
            transR, transT = self.get_transform(p1, p2)
            return self.RT2SE3(transR, transT)

    def RT2SE3(self, transR: torch.tensor, transT: torch.tensor):
        r'''
        Transfers the transR and transT into SE3 Lie tensor

        Args:
            transR:
            transT:

        Returns:
            LieTensor SE3
        '''

        temp_transT = transT.reshape([3, 1])
        T = torch.cat([torch.cat([transR, temp_transT], dim=1), torch.tensor([[0., 0., 0., 1.]], dtype=torch.float)])
        return pp.from_matrix(T, ltype=pp.SE3_type)

    def get_transform(self, p1: torch.tensor, p2: torch.tensor):
        r'''Using SVD algorithm to calculate the transformation matrix between the corresponding points set p1, p2.

        Args:
            p1: input points set
            p2: the matched target points set

        Returns:
            R: the rotation matrix between p1 and p2
            T: the translation matrix between p1 and p2
        '''
        p1_centroid = p1.mean(0)
        p2_centroid = p2.mean(0)

        temp_p1 = (p1 - p1_centroid)
        temp_p2 = (p2 - p2_centroid)

        H = torch.matmul(transpose(temp_p2), temp_p1)
        u, s, vT = torch.linalg.svd(H)
        v = transpose(vT)
        uT = transpose(u)

        if (torch.det(v * uT) < 0):
            v[2, :] *= -1
        R = torch.matmul(u, vT)
        t = p2_centroid - torch.matmul(p1_centroid, transpose(R))

        return R, t


    def nearest_neighbor(self, p1, p2):
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

        distances = torch.zeros([p1.size()[0], 1])
        indices = list(range(p1.size()[0]))
        for i in range(p1.size()[0]):
            dist1 = torch.norm(p2-p1[i],dim=1,p=None)
            nn = dist1.topk(1, largest = False)
            distances[i] = nn.values[0]
            indices[i] = int(nn.indices[0])

        return distances, indices

def transpose(input: torch.tensor):
    r'''
    Transpose the input matrix from nxm to mxn

    Args:
        input: nxm tensor

    Returns:
        output: mxn tensor
    '''

    row_num = input.size()[0]
    col_num = input.size()[1]
    output = torch.zeros([col_num, row_num], dtype=torch.float)

    for i in range(col_num):
        for j in range(row_num):
            output[i][j] = input[j][i]

    return output

if __name__=="__main__":
    input_pc = torch.randn([20, 3], dtype=torch.float)
    transT = torch.tensor([0.12, 0.034, 0.052], dtype=torch.float)
    transR = torch.tensor([[0.7071, -0.7071, 0], [0.7071, 0.7071, 0], [0, 0, 1]], dtype=torch.float)
    output_pc = (input_pc @ transpose(transR)) + transT
    icpsvd = ICP(is_matching=False)
    T = icpsvd(input_pc, output_pc)
    print("The LieTensor of transformation is \n", T)
    print("The transformation matrix is \n", pp.matrix(T))
