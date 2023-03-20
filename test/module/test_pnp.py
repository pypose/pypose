import io, os, urllib
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
    proj_mat = torch.bmm(intrinsics[..., :3], rt)
    # concat 1 to the last column of objPts_w
    obj_pts_w_ex = torch.cat((pts_w, torch.ones_like(pts_w[..., :1])), dim=-1)
    # Calculate the image points
    img_repj = torch.bmm(obj_pts_w_ex, proj_mat.transpose(dim0=-1, dim1=-2))

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
    def test_epnp_5pts(self):
        # TODO: Implement this
        return

    def test_epnp_10pts(self):
        # TODO: Implement this
        return

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

            obj_pts = obj_pts[None].to(torch.float32)
            img_pts = img_pts[None].to(torch.float32)
            intrinsics = intrinsics[None].to(torch.float32)
            Rt = Rt[None]

            rot = Rt[:, :3, :3]
            t = Rt[:, :3, 3]

            error = reprojection_error(obj_pts,
                                       img_pts,
                                       intrinsics,
                                       Rt, )
            return dict(Rt=Rt, error=error, R=rot, T=t)

        data = load_data()

        # instantiate epnp
        epnp = pp.module.EPnP(naive=False)
        solution = epnp(data['objPts'], data['imgPts'], data['camMat'])
        solution_ref = solution_opencv(data['objPts'][0],
                                       data['imgPts'][0],
                                       data['camMat'][0])
        gt_rot = data['Rt'][..., :3, :3]
        gt_t = data['Rt'][..., :3, 3]

        print("Pypose EPnP solution, rmse of R:", rmse_rot(solution.rotation().matrix(), gt_rot))
        print("Pypose EPnP solution, rmse of t:", rmse_t(solution.translation(), gt_t))

        print("OpenCV EPnP solution, rmse of R:", rmse_rot(solution_ref['R'], gt_rot))
        print("OpenCV EPnP solution, rmse of t:", rmse_t(solution_ref['T'], gt_t))


if __name__ == "__main__":
    epnp_fixture = TestEPnP()
    epnp_fixture.test_epnp_5pts()
    epnp_fixture.test_epnp_10pts()
    epnp_fixture.test_epnp_random()