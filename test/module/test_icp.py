import torch
import pypose as pp
from torchvision.datasets.utils import download_and_extract_archive

class TestICP:

    def load_point_cloud(self):
        download_and_extract_archive('https://github.com/Murphy41/laser-scan-pt/' \
                                     'releases/download/v0.0/icp-test-data.pt.zip',\
                                    './test/module')
        loaded_tensors = torch.load('./test/module/icp-test-data.pt')
        pc1 = loaded_tensors['pc1'].squeeze(-3)
        pc2 = loaded_tensors['pc2'].squeeze(-3)
        return pc1, pc2

    # def test_icp_nonbatch(self):
    #     pc1, pc2 = self.load_point_cloud()
    #     tf_gt_mat = torch.tensor([[0.67947332,  0.69882627, 0.22351252, -0.07126862],
    #                             [-0.73099025, 0.61862761, 0.28801584, -0.22845308],
    #                             [0.06300202, -0.35908455, 0.93117615,6.92614964],
    #                             [0, 0, 0, 1]])

    #     # tf_gt = torch.tensor([[0,  -1, 0, -0.05],
    #     #                         [1, 0, 0, -0.02],
    #     #                         [0, 0, 1, 0.03],
    #     #                         [0, 0, 0, 1]])

    #     self.tf_gt = pp.mat2SE3(tf_gt_mat)
    #     pc2 = self.tf_gt.Act(pc2)
    #     icp = pp.module.ICP()
    #     self.result = icp(pc1, pc2)
    #     print("The true tf is", self.tf_gt)
    #     print("The output is", self.result)
    #     self.rmse_results()

    def test_icp_batch(self):
        b = 3
        num_points = 20
        pc1 = torch.rand(b, num_points, 3)
        self.tf_gt = pp.randn_SE3(b)
        pc2 = self.tf_gt.unsqueeze(-2).Act(pc1)
        icp = pp.module.ICP()
        self.result = icp(pc1, pc2)
        print("The true tf is", self.tf_gt)
        print("The output is", self.result)
        self.rmse_results()

    def rmse_results(self):
        rot = self.result.rotation().matrix()
        t = self.result.translation()
        gt_rot = self.tf_gt.rotation().matrix()
        gt_t = self.tf_gt.translation()

        def rmse_rot(pred, gt):
            return torch.linalg.norm((pred - gt), dim=-1, ord=2).mean(-1)

        def rmse_t(pred, gt):
            return torch.linalg.norm((pred - gt), dim=-1, ord=2)

        torch.set_printoptions(precision=2, sci_mode=False)
        print("Pypose ICP solution, rmse of R:", rmse_rot(rot, gt_rot))
        print("Pypose ICP solution, rmse of t:", rmse_t(t, gt_t))



if __name__ == "__main__":
    test = TestICP()
    # test.test_icp_nonbatch()
    test.test_icp_batch()
