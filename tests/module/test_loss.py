import math
import pypose as pp
import torch as torch


class TestLoss:

    def test_geodesic(self):

        x, y = pp.randn_SE3(2), pp.randn_SE3(2)

        criterion = pp.module.GeodesicLoss(reduction='none')
        e1 = criterion(x, y)

        x = x.rotation()
        y = y.rotation()
        e2 = (((x * y.Inv()).matrix().diagonal(dim1=-1, dim2=-2).sum(dim=-1)-1)/2).arccos()

        assert torch.allclose(e1, e2), "GeodesicLoss computation is incorrect."


if __name__ == '__main__':
    test = TestLoss()
    test.test_geodesic()
