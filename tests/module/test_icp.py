import torch
import pypose as pp
from torchvision.datasets.utils import download_and_extract_archive

class TestICP:

    def test_icp_laserscan_data(self):
        # pc1 and pc2 has different numbers of points
        download_and_extract_archive('https://github.com/Murphy41/laser-scan-pt/' \
                                     'releases/download/v0.0/icp-test-data.pt.zip',\
                                     './tests/module')
        loaded_tensors = torch.load('./tests/module/icp-test-data.pt')
        pc1 = loaded_tensors['pc1'].squeeze(-3)
        pc2 = loaded_tensors['pc2'].squeeze(-3)
        tf = pp.SE3([-0.0500, -0.0200,  0.0000, 0, 0, 0.0499792, 0.9987503])
        pc2 = tf.Act(pc2)
        icp = pp.module.ICP()
        result = icp(pc1, pc2)
        error = pp.posediff(tf,result,aggregate=True)
        print("Test 1 (real laser scan data test): The translational error is {:.4f} and "
              "the rotational error is {:.4f}".format(error[0].item(), error[1].item()))
        assert error[0] < 0.1,  "The translational error is too large."
        assert error[1] < 0.1,  "The rotational error is too large."

    def test_icp_batch(self):
        n_points = 1000
        # Generate points on the L shape wall with x = 0:10, y = 20, and x = 10, y = 20:0
        x_line_1 = torch.linspace(0, 10, n_points // 2)
        y_line_1 = torch.zeros(n_points // 2)
        z_line_1 = torch.zeros(n_points // 2)
        x_line_2 = torch.full((n_points // 2,), 10)
        y_line_2 = torch.linspace(20, 0, n_points // 2)
        z_line_2 = torch.zeros(n_points // 2)
        points_set_1 = torch.stack((torch.cat((x_line_1, x_line_2)),
                                    torch.cat((y_line_1, y_line_2)),
                                    torch.cat((z_line_1, z_line_2))), dim=1)
        # Generate points for a curve in the x-y plane with noise
        radius = 10
        noise_std_dev = 1
        theta = torch.linspace(0, 0.5 * 3.14159265, n_points)
        x_curve = 10 + radius * torch.cos(theta) + torch.randn(n_points) * noise_std_dev
        y_curve = 20 - radius * torch.sin(theta) + torch.randn(n_points) * noise_std_dev
        z_curve = torch.zeros(n_points) + torch.randn(n_points) * noise_std_dev
        points_set_2 = torch.stack((x_curve, y_curve, z_curve), dim=1)
        # Test ICP
        pc1 = torch.stack((points_set_1, points_set_2), dim=0)
        tf = pp.SE3([[-5.05, -3.02,  0.02, 0, 0, 0.0499792, 0.9987503],
               [-2, 1, 1, 0.1304815, 0.0034168, -0.025953, 0.9911051]])
        pc2 = tf.unsqueeze(-2).Act(pc1)
        icp = pp.module.ICP()
        result = icp(pc1, pc2)
        error = pp.posediff(tf,result,aggregate=True)
        print("Test 2 (batched generated data test): The translational error is {:.4f} "
              "and the rotational error is {:.4f}"
              .format(error[0].item(), error[1].item()))
        assert error[0] < 0.1,  "The translational error is too large."
        assert error[1] < 0.1,  "The rotational error is too large."

if __name__ == "__main__":
    torch.set_printoptions(precision=4, sci_mode=False)
    test = TestICP()
    test.test_icp_laserscan_data()
    test.test_icp_batch()
