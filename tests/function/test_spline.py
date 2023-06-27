import torch
import pypose as pp

class TestSpline:

    def test_bsplilne(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # test two poses
        timeline = torch.arange(0, 1, 0.1, device=device)
        data = pp.randn_SE3(2,device=device)
        poses = pp.bspline(data, timeline)
        torch.testing.assert_close(poses[...,[0,-1],:].translation(),
                                   data[...,[0,-1],:].translation(),atol=1e-4, rtol=1e-2)
        torch.testing.assert_close(poses[...,[0,-1],:].rotation(),
                                   data[...,[0,-1],:].rotation(),atol=1e-4, rtol=1e-2)
        # test for multi batch
        data = pp.randn_SE3(2,3, device=device)
        poses = pp.bspline(data, timeline)
        torch.testing.assert_close(poses[...,[0,-1],:].translation(),
                                   data[...,[0,-1],:].translation(),atol=1e-4, rtol=1e-2)
        torch.testing.assert_close(poses[...,[0,-1],:].rotation(),
                                   data[...,[0,-1],:].rotation(),atol=1e-4, rtol=1e-2)
        # test for high dimension
        data = pp.randn_SE3(2,3,4, device=device)
        poses = pp.bspline(data, timeline)
        torch.testing.assert_close(poses[...,[0,-1],:].translation(),
                                   data[...,[0,-1],:].translation(),atol=1e-4, rtol=1e-2)
        torch.testing.assert_close(poses[...,[0,-1],:].rotation(),
                                   data[...,[0,-1],:].rotation(),atol=1e-4, rtol=1e-2)

    def test_chspline(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # test for different point dimension
        points = 10*torch.randn(1,2,3, device=device)
        interpoints = pp.chspline(points, interval=0.1)
        num = points.shape[-2]
        po = interpoints.index_select(-2, torch.arange(0, int(num/0.1),
                                                       1/0.1, device=device).int())
        assert (points - po).sum()<1e-5
        # test multi points
        points = torch.randn(20, 3, device=device)
        interpoints = pp.chspline(points, interval=0.2)
        num = points.shape[-2]
        po = interpoints.index_select(-2, torch.arange(0, int(num/0.2),
                                                       1/0.2, device=device).int())
        assert (points - po).sum()<1e-5
        # test multi batches
        points = torch.randn(3,20,3, device=device)
        interpoints = pp.chspline(points, interval=0.25)
        num = points.shape[-2]
        po = interpoints.index_select(-2, torch.arange(0, int(num/0.25),
                                                       1/0.25,device=device).int())
        assert (points - po).sum()<1e-5
        # test multi dim of points
        points = torch.randn(2,3,50,4, device=device)
        interpoints = pp.chspline(points)
        num = points.shape[-2]
        po = interpoints.index_select(-2, torch.arange(0, int(num/0.1),
                                                       1/0.1, device=device).int())
        assert (points - po).sum()<1e-5
        points = torch.randn(10,2,3,50,4, device=device)
        interpoints = pp.chspline(points)
        num = points.shape[-2]
        po = interpoints.index_select(-2, torch.arange(0, int(num/0.1),
                                                       1/0.1, device=device).int())
        assert (points - po).sum()<1e-5


if __name__=="__main__":
    spline = TestSpline()
    spline.test_bsplilne()
    spline.test_chspline()
