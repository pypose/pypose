import torch, pypose as pp
from torchvision.datasets.utils import download_and_extract_archive


class TestEPnP:

    def load_data(self):
        download_and_extract_archive('https://github.com/pypose/pypose/releases/'\
                                     'download/v0.3.6/epnp-test-data.pt.zip', \
                                     './tests/module')
        return torch.load('./tests/module/epnp-test-data.pt')

    def test_epnp_nonbatch(self):
        data = self.load_data()
        epnp = pp.module.EPnP()
        points = data['objPts'].unsqueeze(0)
        pixels = data['imgPts'].unsqueeze(0)
        intrincis = data['camMat'].unsqueeze(0)
        pose_non_batch = epnp(points, pixels, intrincis)
        pose_batch = epnp(data['objPts'], data['imgPts'], data['camMat'])
        assert pose_non_batch[0].shape == pose_batch.shape == torch.Size([2, 7])
        assert torch.allclose(pose_non_batch, pose_batch[0])

    def test_epnp_highdim(self):
        data = self.load_data()
        epnp = pp.module.EPnP()
        # batch shape: [3, 2, ...]
        points = data['objPts'][None][[0, 0, 0]]
        pixels = data['imgPts'][None][[0, 0, 0]]
        intrincis = data['camMat'][None][[0, 0, 0]]
        solution_highdim = epnp(points, pixels, intrincis)
        solution_lowdim = epnp(data['objPts'], data['imgPts'], data['camMat'])
        assert solution_highdim[0].shape == solution_lowdim.shape
        assert torch.allclose(solution_highdim[0], solution_lowdim)

    def test_epnp_6pts(self):
        # create some random test sample for a single camera
        pose = pp.SE3([ 0.0000, -8.0000,  0.0000,
                        0.0000, -0.3827,  0.0000,  0.9239]).to(torch.float64)
        f, img_size = 2, (9, 9)
        projection = torch.tensor([[f, 0, img_size[0] / 2],
                                   [0, f, img_size[1] / 2],
                                   [0, 0, 1              ]], dtype=torch.float64)
        # some random points in the view
        pts_c = torch.tensor([[2., 0., 2.],
                              [1., 0., 2.],
                              [0., 1., 1.],
                              [0., 0., 1.],
                              [1., 0., 1.],
                              [5., 5., 3.]], dtype=torch.float64)
        pixels = pp.homo2cart(pts_c @ projection.T)
        # transform the points to world coordinate
        # solve the PnP problem to find the camera pose
        pts_w = pose.Inv().Act(pts_c)
        # solve the PnP problem
        epnp = pp.module.EPnP(intrinsics=projection)
        solved_pose = epnp(pts_w, pixels)
        torch.testing.assert_close(solved_pose, pose, rtol=1e-4, atol=1e-4)

    def test_epnp_random(self):
        # results given by cv2.solvePnP(obj_pts, img_pts, intrinsics, \
        #                               distortion, flags=cv2.SOLVEPNP_EPNP)
        t_ref = torch.tensor([-0.07126862, -0.22845308, 6.92614964])
        rot_ref = torch.tensor([[0.67947332,  0.69882627, 0.22351252],
                                [-0.73099025, 0.61862761, 0.28801584],
                                [0.06300202, -0.35908455, 0.93117615]])

        data = self.load_data()
        epnp = pp.module.EPnP()
        pose = epnp(data['objPts'], data['imgPts'], data['camMat'])
        rot = pose.rotation().matrix()
        t = pose.translation()
        gt_rot = data['Rt'][..., :3, :3]
        gt_t = data['Rt'][..., :3, 3]

        def rmse_rot(pred, gt):
            return (pred - gt).norm(dim=(-2, -1)).mean()

        def rmse_t(pred, gt):
            return (pred - gt).pow(2).sum(dim=-1).mean()

        print("Pypose EPnP solution, rmse of R:", rmse_rot(rot, gt_rot))
        print("Pypose EPnP solution, rmse of t:", rmse_t(t, gt_t))
        print("OpenCV EPnP solution, rmse of R:", rmse_rot(rot_ref, gt_rot))
        print("OpenCV EPnP solution, rmse of t:", rmse_t(t_ref, gt_t))


if __name__ == "__main__":
    epnp_fixture = TestEPnP()
    epnp_fixture.test_epnp_nonbatch()
    epnp_fixture.test_epnp_highdim()
    epnp_fixture.test_epnp_6pts()
    epnp_fixture.test_epnp_random()
