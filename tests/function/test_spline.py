import math
import torch
import pypose as pp

class TestSpline:

    def test_bsplilne(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # test two poses
        data = pp.randn_SE3(2,device=device)
        poses = pp.bspline(data, 0.10, True)
        pp.testing.assert_close(poses[...,[0,-1],:], data[...,[0,-1],:])

        # test for multi batch
        data = pp.randn_SE3(2,5, device=device)
        poses = pp.bspline(data, 0.5)
        assert poses.lshape[-1] == 2 * (data.lshape[-1]-3)+1
        poses = pp.bspline(data, 0.5, True)
        pp.testing.assert_close(poses[...,[0,-1],:], data[...,[0,-1],:])

        # test for high dimension
        data = pp.randn_SE3(2,3,4, device=device)
        poses = pp.bspline(data, 0.2)
        assert poses.lshape[-1] == 5 * (data.lshape[-1]-3)+1
        poses = pp.bspline(data, 0.2, True)
        pp.testing.assert_close(poses[...,[0,-1],:], data[...,[0,-1],:])

        data = pp.randn_SE3(2,3,4, device=device)
        poses = pp.bspline(data, 0.3)
        assert poses.lshape[-1] == 4 * (data.lshape[-1]-3)+1
        poses = pp.bspline(data, 0.3, True)
        pp.testing.assert_close(poses[...,[0,-1],:], data[...,[0,-1],:])

    def test_chspline(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # test for different point dimension
        points = torch.randn(1,2,3, device=device)
        interval = .2
        interpoints = pp.chspline(points, interval=interval)
        num = points.shape[-2]
        k = math.ceil(1.0 / interval)
        index = k*torch.arange(0, num, device=device, dtype=torch.int64)
        po = interpoints.index_select(-2, index)
        pp.testing.assert_close(points, po)

        # test multi points
        points = torch.randn(20, 3, device=device)
        interval = 0.5
        interpoints = pp.chspline(points, interval=interval)
        num = points.shape[-2]
        k = math.ceil(1.0 / interval)
        index = k*torch.arange(0, num, device=device, dtype=torch.int64)
        po = interpoints.index_select(-2, index)
        pp.testing.assert_close(points, po)

        # test multi batches
        points = torch.randn(3,20,3, device=device)
        interval = 0.4
        interpoints = pp.chspline(points, interval=interval)
        num = points.shape[-2]
        k = math.ceil(1.0/interval)
        index = k*torch.arange(0, num, device=device, dtype=torch.int64)
        po = interpoints.index_select(-2, index)
        pp.testing.assert_close(points, po)

        # test multi dim of points
        points = torch.randn(2,3,50,4, device=device)
        interval = 0.1
        interpoints = pp.chspline(points, interval=interval)
        num = points.shape[-2]
        k = math.ceil(1.0 / interval)
        index = k*torch.arange(0, num, device=device, dtype=torch.int64)
        po = interpoints.index_select(-2, index)
        pp.testing.assert_close(points, po)

        points = torch.randn(10,2,3,50,4, device=device)
        interval = 0.1
        interpoints = pp.chspline(points, interval=interval)
        num = points.shape[-2]
        k = math.ceil(1.0 / interval)
        index = k*torch.arange(0, num, device=device, dtype=torch.int64)
        po = interpoints.index_select(-2, index)
        pp.testing.assert_close(points, po)


if __name__=="__main__":
    spline = TestSpline()
    spline.test_bsplilne()
    spline.test_chspline()
