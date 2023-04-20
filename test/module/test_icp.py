import torch
import pypose as pp
from torchvision.datasets.utils import download_and_extract_archive

class TestICP:

    def load_point_cloud(self):
        download_and_extract_archive('https://github.com/Murphy41/laser-scan-pt/' \
                                     'releases/download/v0.0/icp-test-data.pt.zip',\
                                    './test/module')
        loaded_tensors = torch.load('./test/module/icp-test-data.pt')
        pc1 = loaded_tensors['pc1']
        pc2 = loaded_tensors['pc2']
        return pc1, pc2

    def test_laser_scan(self):
        pc1, pc2 = self.load_point_cloud()
        tf = pp.randn_SE3(1)
        pc2 = tf.unsqueeze(-2).Act(pc2)
        icpsvd = pp.module.ICP()
        result = icpsvd(pc1, pc2)
        print("The true tf is", tf)
        print("The output is", result)

    def test_random_pc(self):
        b = 3
        num_points = 20
        pc1 = torch.rand(b, num_points, 3)
        tf = pp.randn_SE3(b)
        pc2 = tf.unsqueeze(-2).Act(pc1)
        icpsvd = pp.module.ICP()
        result = icpsvd(pc1, pc2)
        print("The true tf is", tf)
        print("The output is", result)

if __name__ == "__main__":
    test = TestICP()
    test.test_random_pc()
    test.test_laser_scan()
