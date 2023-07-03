import torch
import pypose as pp

class TestSpline:

    def test_bsplilne(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # test two poses
        data = pp.randn_SE3(2,device=device)
        poses = pp.bspline(data, 10)
        assert poses.lshape[-1] == 10 * (data.lshape[-1]-1)
        poses = pp.bspline(data, 10, True)
        torch.testing.assert_close(poses[...,[0,-1],:].translation(),
                                   data[...,[0,-1],:].translation())
        torch.testing.assert_close(poses[...,[0,-1],:].rotation(),
                                   data[...,[0,-1],:].rotation())
        # test for multi batch
        data = pp.randn_SE3(2,3, device=device)
        poses = pp.bspline(data, 5)
        assert poses.lshape[-1] == 5 * (data.lshape[-1]-1)
        poses = pp.bspline(data, 5, True)
        torch.testing.assert_close(poses[...,[0,-1],:].translation(),
                                   data[...,[0,-1],:].translation())
        torch.testing.assert_close(poses[...,[0,-1],:].rotation(),
                                   data[...,[0,-1],:].rotation())
        # test for high dimension
        data = pp.randn_SE3(2,3,4, device=device)
        poses = pp.bspline(data, 20)
        assert poses.lshape[-1] == 20 * (data.lshape[-1]-1)
        poses = pp.bspline(data, 20, True)
        torch.testing.assert_close(poses[...,[0,-1],:].translation(),
                                   data[...,[0,-1],:].translation())
        torch.testing.assert_close(poses[...,[0,-1],:].rotation(),
                                   data[...,[0,-1],:].rotation())

    def test_chspline(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # test for different point dimension
        points = torch.randn(1,2,3, device=device)
        interpoints = pp.chspline(points, 10)
        num = points.shape[-2]
        index = torch.arange(0, int(num/0.1), 1/0.1, device=device, dtype=torch.int64)
        po = interpoints.index_select(-2, index)
        assert (points - po).sum()<1e-5
        # test multi points
        points = torch.randn(20, 3, device=device)
        interpoints = pp.chspline(points, 5)
        num = points.shape[-2]
        index = torch.arange(0, int(num/0.2), 1/0.2, device=device, dtype=torch.int64)
        po = interpoints.index_select(-2, index)
        assert (points - po).sum()<1e-5
        # test multi batches
        points = torch.randn(3,20,3, device=device)
        interpoints = pp.chspline(points, 4)
        num = points.shape[-2]
        index = torch.arange(0, int(num/0.25), 1/0.25, device=device, dtype=torch.int64)
        po = interpoints.index_select(-2, index)
        assert (points - po).sum()<1e-5
        # test multi dim of points
        points = torch.randn(2,3,50,4, device=device)
        interpoints = pp.chspline(points)
        num = points.shape[-2]
        index = torch.arange(0, int(num/0.1), 1/0.1, device=device, dtype=torch.int64)
        po = interpoints.index_select(-2, index)
        assert (points - po).sum()<1e-5
        points = torch.randn(10,2,3,50,4, device=device)
        interpoints = pp.chspline(points)
        num = points.shape[-2]
        index = torch.arange(0, int(num/0.1), 1/0.1, device=device, dtype=torch.int64)
        po = interpoints.index_select(-2, index)
        assert (points - po).sum()<1e-5


if __name__=="__main__":
    spline = TestSpline()
    spline.test_bsplilne()
    spline.test_chspline()
