import torch
import pypose as pp
from torchvision.datasets.utils import download_and_extract_archive

class TestICP:

    def load_data(self):
        # pc1 and pc2 has different numbers of points
        download_and_extract_archive('https://github.com/pypose/pypose/releases/'\
                                     'download/v0.4.2/icp-test-data.pt.zip',\
                                     './tests/module')
        loaded_tensors = torch.load('./tests/module/icp-test-data.pt')
        self.pc1 = loaded_tensors['pc1'].squeeze(-3)
        self.pc2 = loaded_tensors['pc2'].squeeze(-3)

    def test_icp_laserscan_data(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_data()
        source = self.pc1.to(device)
        tf = pp.SE3([-0.0500, -0.0200,  0.0000, 0, 0, 0.0499792, 0.9987503])
        target = tf.Act(self.pc2).to(device)
        icp = pp.module.ICP().to(device)
        result = icp(source, target)
        error = _posediff(tf.to(device),result,aggregate=True)
        print("Test 1 (real laser scan data test): The translational error is {:.4f} and "
              "the rotational error is {:.4f}".format(error[0].item(), error[1].item()))
        assert error[0] < 0.1,  "The translational error is too large."
        assert error[1] < 0.1,  "The rotational error is too large."

    def test_icp_batch(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        source = torch.stack((points_set_1, points_set_2), dim=0).to(device)
        tf = pp.SE3([[-5.05, -3.02,  0.02,         0,         0, 0.0499792, 0.9987503],
                     [   -2,      1,    1, 0.1304815, 0.0034168, -0.025953, 0.9911051]]).to(device)
        target = tf.unsqueeze(-2).Act(source).to(device)
        icp = pp.module.ICP().to(device)
        result = icp(source, target)
        error = _posediff(tf,result,aggregate=True)
        print("Test 2 (batched generated data test): The translational error is {:.4f} "
              "and the rotational error is {:.4f}"
              .format(error[0].item(), error[1].item()))
        assert error[0] < 0.1,  "The translational error is too large."
        assert error[1] < 0.1,  "The rotational error is too large."

    def test_icp_broadcasting1(self):
        self.load_data()
        source = self.pc1
        tf = pp.SE3([[-0.0500, -0.0200,  0.0000, 0, 0, 0.0499792, 0.9987503],
                     [-0.0500, -0.0200,  0.0000, 0, 0, 0.0499792, 0.9987503]])
        target = tf.unsqueeze(-2).Act(self.pc2)
        icp = pp.module.ICP()
        result = icp(source, target)
        error = _posediff(tf,result,aggregate=True)
        print("Test 3 (broadcasting test 1): The translational error is {:.4f} "
              "and the rotational error is {:.4f}"
              .format(error[0].item(), error[1].item()))
        assert error[0] < 0.1,  "The translational error is too large."
        assert error[1] < 0.1,  "The rotational error is too large."

    def test_icp_broadcasting2(self):
        self.load_data()
        target = self.pc2
        tf = pp.SE3([[-0.0500, -0.0200,  0.0000, 0, 0, 0.0499792, 0.9987503],
                     [-0.0100, -0.0300,  0.0000, 0, 0, 0.0499792, 0.9987503]])
        source = tf.unsqueeze(-2).Act(self.pc1)
        stepper = pp.utils.ReduceToBason(steps=100, patience=3, verbose=True)
        torch.set_printoptions(precision=7)
        icp = pp.module.ICP(stepper=stepper)
        result = icp(source, target)
        error = _posediff(tf.Inv(),result,aggregate=True)
        print("Test 4 (broadcasting test 2): The translational error is {:.4f} "
              "and the rotational error is {:.4f}"
              .format(error[0].item(), error[1].item()))
        assert error[0] < 0.1,  "The translational error is too large."
        assert error[1] < 0.1,  "The rotational error is too large."

def _posediff(ref, est, aggregate=False, mode=1):
    r'''
    Computes the translatinal and rotational error between two batched transformations
    ( :math:`SE(3)` ).

    Args:
        ref (``LieTensor``): The reference transformation :math:`T_{ref}` in
            ``SE3type``. The shape is [..., 7].
        est (``LieTensor``): The estimated transformation :math:`T_{est}` in
            ``SE3type``. The shape is [..., 7].
        aggregate (``bool``, optional): Average the batched differences to a singleton
            dimension. Default: ``False``.
        mode (``int``, optional): Calculate the rotational difference in different mode.
            ``mode = 0``: Quaternions representation.
            ``mode = 1``: Axis-angle representation (Use one angle to represent the
            rotational difference in 3D space). Default: ``1``.

    Note:
        The rotation matrix to axis-angle representation refers to the theorem 2.5 and
        2.6 in Chapter 2 [1]. The implementation of the Quaternions to axis-angle
        representation (equation: :math:`\theta = 2 \cos^{-1}(q_0)` ) is presented at
        the end of Chapter 2 in [1].

        [1] Murray, R. M., Li, Z., & Sastry, S. S. (1994). A mathematical introduction to
        robotic manipulation. CRC press.


    Returns:
        ``torch.Tensor``: The translational difference (:math:`\Delta t`) and rotational
        differences between two sets of transformations.

        If ``aggregate = True``: The output batch will be 1.

        If ``mode = 0``: The values in each batch is :math:`[ \Delta t, \Delta q_x,
        \Delta q_y, \Delta q_z, \Delta q_w ]`

        If ``mode = 1``: The values in each batch is :math:`[ \Delta t, \Delta \theta ]`

    Example:
        >>> import torch, pypose as pp
        >>> ref = pp.randn_SE3(4)
        >>> est = pp.randn_SE3(4)
        >>> pp.posediff(ref,est)
        tensor([[3.1877, 0.3945],
        [3.3388, 2.0563],
        [2.4523, 0.4169],
        [1.8392, 1.1539]])
        >>> pp.posediff(ref,est,aggregate=True)
        tensor([1.9840, 1.9306])
        >>> pp.posediff(ref,est,mode=0)
        tensor([[ 3.1877,  0.1554,  0.1179, -0.0190,  0.9806],
        [ 3.3388, -0.0194, -0.8539,  0.0609,  0.5164],
        [ 2.4523,  0.0495, -0.1739,  0.1006,  0.9784],
        [ 1.8392, -0.5451, -0.0075,  0.0192,  0.8381]])
    '''
    assert pp.is_SE3(ref), "The input reference transformation is not SE3Type."
    assert pp.is_SE3(est), "The input estimated transformation is not SE3Type."
    assert mode in (0, 1), "Mode number is invalid."
    T = ref * est.Inv()
    diff_t = torch.linalg.norm(T.translation(), dim=-1, ord=2).unsqueeze(-1)
    if mode == 0:
        diff_r = T.rotation().tensor()
        diff = torch.cat((diff_t, diff_r), dim=-1)
    else:
        qw = torch.clamp(T.tensor()[...,6], min=-1, max=1) # floating-point inaccuracies
        diff_r = 2 * torch.acos(qw)
        diff = torch.cat((diff_t, diff_r.unsqueeze(-1)), dim=-1)
    if aggregate and diff.ndim > 1:
        diff = diff.mean(dim=tuple(range(diff.ndim - 1)), keepdim=True).flatten()
    return diff


if __name__ == "__main__":
    torch.set_printoptions(precision=4, sci_mode=False)
    test = TestICP()
    test.test_icp_laserscan_data()
    test.test_icp_batch()
    test.test_icp_broadcasting1()
    test.test_icp_broadcasting2()
