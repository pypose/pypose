import torch, pypose as pp
from torchvision.datasets.utils import download_and_extract_archive


def reprojection_error(pts_w, img_pts, intrinsics, rt):
    """
    Calculate the reprojection error.
    Args:
        pts_w: The object points in world coordinate. The shape is (..., N, 3).
        img_pts: The image points. The shape is (..., N, 2).
        intrinsics: The camera matrix. The shape is (..., 3, 3).
        rt: The rotation matrix and translation vector. The shape is (..., 3, 4).
    Returns:
        error: The reprojection error. The shape is (..., ).
    """
    proj_mat = intrinsics[..., :3] @ rt
    # concat 1 to the last column of objPts_w
    obj_pts_w_ex = torch.cat((pts_w, torch.ones_like(pts_w[..., :1])), dim=-1)
    # Calculate the image points
    img_repj = obj_pts_w_ex @ proj_mat.mT

    # Normalize the image points
    img_repj = img_repj[..., :2] / img_repj[..., 2:]

    error = torch.linalg.norm(img_repj - img_pts, dim=-1)
    error = torch.mean(error, dim=-1)

    return error


def load_data():
    download_and_extract_archive('https://github.com/pypose/pypose/releases/'\
                                 'download/v0.3.6/epnp-test-data.pt.zip', '.')
    return torch.load('./epnp-test-data.pt')


def rmse_rot(pred, gt):
    diff = pred - gt
    f_norm = torch.norm(diff, dim=(-2, -1))
    return f_norm.mean()


def rmse_t(pred, gt):
    diff = pred - gt
    norm = diff ** 2
    norm = torch.sum(norm, dim=-1)
    return norm.mean()


class TestEPnP:
    def test_epnp_nonbatch(self):
        data = load_data()
        epnp = pp.module.EPnP()
        solution_non_batch = epnp(data['objPts'][0], data['imgPts'][0], data['camMat'][0])
        solution_batch = epnp(data['objPts'], data['imgPts'], data['camMat'])
        assert torch.allclose(solution_non_batch.rotation().matrix(), solution_batch.rotation().matrix()[0])


    def test_epnp_highdim(self):
        data = load_data()
        epnp = pp.module.EPnP()
        # batch shape: [3, 2, ...]
        solution_highdim = epnp(data['objPts'][None][[0, 0, 0]], data['imgPts'][None][[0, 0, 0]], data['camMat'][None][[0, 0, 0]])
        # batch shape: [2, ...]
        solution_lowdim = epnp(data['objPts'], data['imgPts'], data['camMat'])

        assert solution_highdim.rotation().matrix()[0].shape == solution_lowdim.rotation().matrix().shape
        assert torch.allclose(solution_highdim.rotation().matrix()[0], solution_lowdim.rotation().matrix())


    def test_epnp_6pts(self):
        # create some random test sample for a single camera
        pose = pp.SE3([ 0.0000, -8.0000,  0.0000,  0.0000, -0.3827,  0.0000,  0.9239]).to(torch.float64)
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
        def solution_opencv(obj_pts, img_pts, intrinsics):
            distortion = None
            # results given by cv2.solvePnP(obj_pts, img_pts, intrinsics, distortion, flags=cv2.SOLVEPNP_EPNP)
            rvec = torch.tensor([[-0.37279579],
                                 [0.09247041],
                                 [-0.82372009]])
            t = torch.tensor([[-0.07126862],
                              [-0.22845308],
                              [6.92614964]])
            rot = torch.tensor([[0.67947332, 0.69882627, 0.22351252],
                                [-0.73099025, 0.61862761, 0.28801584],
                                [0.06300202, -0.35908455, 0.93117615]])
            Rt = torch.concatenate((rot.reshape((3, 3)), t.reshape((3, 1))), dim=1)

            obj_pts = obj_pts.to(torch.float32)
            img_pts = img_pts.to(torch.float32)
            intrinsics = intrinsics.to(torch.float32)

            rot = Rt[..., :3, :3]
            t = Rt[..., :3, 3]

            error = reprojection_error(obj_pts, img_pts, intrinsics, Rt)
            return dict(Rt=Rt, error=error, R=rot, T=t)

        data = load_data()

        epnp = pp.module.EPnP()
        solution = epnp(data['objPts'], data['imgPts'], data['camMat'])
        solution_ref = solution_opencv(data['objPts'][0], data['imgPts'][0], data['camMat'][0])
        gt_rot = data['Rt'][..., :3, :3]
        gt_t = data['Rt'][..., :3, 3]

        print("Pypose EPnP solution, rmse of R:", rmse_rot(solution.rotation().matrix(), gt_rot))
        print("Pypose EPnP solution, rmse of t:", rmse_t(solution.translation(), gt_t))

        print("OpenCV EPnP solution, rmse of R:", rmse_rot(solution_ref['R'], gt_rot))
        print("OpenCV EPnP solution, rmse of t:", rmse_t(solution_ref['T'], gt_t))


if __name__ == "__main__":
    epnp_fixture = TestEPnP()
    epnp_fixture.test_epnp_nonbatch()
    epnp_fixture.test_epnp_6pts()
    epnp_fixture.test_epnp_random()
