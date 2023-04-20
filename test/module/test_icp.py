import torch
import pypose as pp
from torchvision.datasets.utils import download_and_extract_archive

class TestICP:

    def load_point_cloud():
        download_and_extract_archive('https://github.com/Murphy41/laser-scan-pt/' \
                                     'releases/download/v0.0/icp-test-data.zip',\
                                    './test/module')
        loaded_tensors = torch.load('./test/module/icp-test-data.pt')
        pc1 = loaded_tensors['pc1']
        pc2 = loaded_tensors['pc2']
        return pc1, pc2

    def laser_scan():
        pc1, pc2 = TestICP.load_point_cloud()
        tf = pp.randn_SE3(1)
        pc2 = tf.unsqueeze(-2).Act(pc2)
        return pc1, pc2, tf

    def random_point_cloud(b, num_points):
        pc1 = torch.rand(b, num_points, 3)
        tf = pp.randn_SE3(b)
        pc2 = tf.unsqueeze(-2).Act(pc1)
        return pc1, pc2, tf



if __name__ == "__main__":
    b = 2
    num_points = 20
    pc1, pc2, tf  = TestICP.random_point_cloud(b, num_points)
    # pc1, pc2, tf = TestICP.laser_scan()
    icpsvd = pp.module.ICP()
    result = icpsvd(pc1, pc2)
    print("The true tf is", tf)
    print("The output is", result)


    # import torch, pypose as pp
    # b = torch.randint(low=1, high=10, size=())
    # num_points1 = torch.randint(low=2, high=100, size=())
    # num_points2 = torch.randint(low=2, high=100, size=())
    # pc1 = torch.rand(b, num_points1, 3)
    # pc2 = torch.rand(b, num_points2, 3)
    # print(pc1.shape)
    # print(pc2.shape)
    # dist, idx = pp.module.ICP._k_nearest_neighbor(pc1, pc2, k = 5, sort = True)
    # print(dist.shape)
    # print(idx.shape)
