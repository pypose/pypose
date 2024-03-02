import math
import torch
import pypose as pp

class TestDownsample:

    def test_random(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # test multiple dimensions
        points = torch.randn(100,2,2, device=device)
        # print(points)
        result = pp.geometry.random_filter(points, 10)
        for p in result:
            assert p in points

    def test_voxel(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # test multiple dimensions
        points = torch.randn(100,2,2, device=device)
        result = pp.geometry.voxel_filter(points, [1., 1.], dim=2)
        # test voxel size larger than total num of point cloud
        points = torch.randn(100,2,3, device=device)
        result = pp.geometry.voxel_filter(points, [1., 1., 1.])
        # test random select with




if __name__=="__main__":
    spline = TestDownsample()
    spline.test_randomfilter()
    spline.test_voxel()
