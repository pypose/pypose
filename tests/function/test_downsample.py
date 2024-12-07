import math
import torch
import pypose as pp

class TestDownsample:

    def test_random(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # test multiple dimensions
        points = torch.randn(10,3)
        num = 5
        result = pp.geometry.random_filter(points, num)
        assert result.size(-2) == num
        for p in result:
            assert p in points

        points = torch.randn(10,2)
        num = 5
        result = pp.geometry.random_filter(points, num)
        assert result.size(-2) == num
        for p in result:
            assert p in points

    def test_voxel(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # test multiple dimensions
        points = torch.tensor([[1., 2.],
                        [4., 5.],
                        [7., 8.],
                        [10., 11.],
                        [13., 14.]])
        result = pp.geometry.voxel_filter(points, [5., 5.], dim=2)
        expect = torch.tensor([[ 2.5000,  3.5000],
                    [ 8.5000,  9.5000],
                    [13.0000, 14.0000]])
        pp.testing.assert_close(result, expect)

        points = torch.tensor([[0., 0., 0.],
                            [1., 0., 0.],
                            [0., 1., 0.],
                            [0., 1., 1.],
                            [10., 1., 1.],
                            [10., 1., 10.]])
        expect = torch.tensor([[ 0.,  0.,  0.],
                    [ 0.,  1.,  0.],
                    [ 0.,  1.,  1.],
                    [ 1.,  0.,  0.],
                    [10.,  1.,  1.],
                    [10.,  1., 10.]])
        result = pp.geometry.voxel_filter(points, [1., 1., 1.])
        pp.testing.assert_close(result, expect)
        # test random select
        points = torch.tensor([[0., 0., 0.],
                            [1., 0., 0.],
                            [0., 1., 0.],
                            [0., 1., 1.],
                            [10., 1., 1.],
                            [10., 1., 10.]])
        result = pp.geometry.voxel_filter(points, [1., 1., 1.])
        for p in result:
            assert p in points



if __name__=="__main__":
    spline = TestDownsample()
    spline.test_random()
    spline.test_voxel()
