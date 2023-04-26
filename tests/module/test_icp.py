import torch
import pypose as pp
from torchvision.datasets.utils import download_and_extract_archive

class TestICP:

    def load_point_cloud(self):
        download_and_extract_archive('https://github.com/Murphy41/laser-scan-pt/' \
                                     'releases/download/v0.0/icp-test-data.pt.zip',\
                                     './tests/module')
        loaded_tensors = torch.load('./tests/module/icp-test-data.pt')
        pc1 = loaded_tensors['pc1'].squeeze(-3)
        pc2 = loaded_tensors['pc2'].squeeze(-3)
        return pc1, pc2

    def test_icp_laserscan(self):
        pc1, pc2 = self.load_point_cloud()
        tf_gt_mat = torch.tensor([[0.9950042, -0.0998334,  0.0000000, -0.05],
                                [0.0998334,  0.9950042,  0.0000000, -0.02],
                                [0.0000000,  0.0000000,  1.0000000, 0],
                                [0, 0, 0, 1]])
        self.tf_gt = pp.mat2SE3(tf_gt_mat)
        pc2 = self.tf_gt.Act(pc2)
        icp = pp.module.ICP()
        self.result = icp(pc1, pc2)
        error = pp.posediff(self.tf_gt,self.result,aggregate=True)
        assert error[0] < 0.01,  "The translational error is too large."
        assert error[1] < 0.01,  "The rotational error is too large."

    def test_icp_batch(self):
        b = 2
        num_points = 2
        pc1 = torch.rand(b, num_points, 3)
        self.tf_gt = pp.randn_SE3(b)
        pc2 = self.tf_gt.unsqueeze(-2).Act(pc1)
        icp = pp.module.ICP()
        self.result = icp(pc1, pc2)
        print(pp.posediff(self.tf_gt,self.result,aggregate=True))

    def test_icp_multibatch(self):
        b1 = 5
        b2 = 3
        num_points = 20
        pc1 = torch.rand(b1, b2, num_points, 3)
        self.tf_gt = pp.randn_SE3(b1,b2)
        pc2 = self.tf_gt.unsqueeze(-2).Act(pc1)
        icp = pp.module.ICP()
        self.result = icp(pc1, pc2)
        self.rmse_results()

if __name__ == "__main__":
    torch.set_printoptions(precision=4, sci_mode=False)
    test = TestICP()
    test.test_icp_laserscan()
    test.test_icp_batch()
    test.test_icp_multibatch()
