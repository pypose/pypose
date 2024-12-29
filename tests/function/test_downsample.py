import torch
import pypose as pp


class TestDownsample:

    def test_random(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # test multiple dimensions
        B, N, num, D = 5, 10, 5, 3
        points = torch.randn(B, N, D, device=device)
        result = pp.random_filter(points, num)
        assert result.shape == torch.Size([B, num, D])

        B, N, num, D = 5, 10, 5, 2
        points = torch.randn(B, N, D, device=device)
        result = pp.random_filter(points, num)
        assert result.shape == torch.Size([B, num, D])


    def test_voxel(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # test multiple dimensions
        points = torch.tensor([[ 1.,  2.],
                               [ 4.,  5.],
                               [ 7.,  8.],
                               [10., 11.],
                               [13., 14.]], device=device)
        result = pp.voxel_filter(points, [5., 5.])
        expect = torch.tensor([[ 2.5000,  3.5000],
                               [ 8.5000,  9.5000],
                               [13.0000, 14.0000]], device=device)
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
        result = pp.voxel_filter(points, [1., 1., 1.])
        pp.testing.assert_close(result, expect)
        # test random select
        points = torch.tensor([[0., 0., 0.],
                            [1., 0., 0.],
                            [0., 1., 0.],
                            [0., 1., 1.],
                            [10., 1., 1.],
                            [10., 1., 10.]])
        result = pp.voxel_filter(points, [1., 1., 1.])
        for p in result:
            assert p in points

        points = torch.tensor([[ 0., 0.,  0., 0],
                               [ 1., 0.,  0., 0],
                               [ 0., 1.,  0., 0],
                               [ 0., 1.,  1., 0],
                               [10., 1.,  1., 0],
                               [10., 1., 10., 0]])
        expect = torch.tensor([[ 0.,  0.,  0., 0],
                               [ 0.,  1.,  0., 0],
                               [ 0.,  1.,  1., 0],
                               [ 1.,  0.,  0., 0],
                               [10.,  1.,  1., 0],
                               [10.,  1., 10., 0]])
        result = pp.voxel_filter(points, [1., 1., 1.])
        pp.testing.assert_close(result, expect)

        # test multiple dimensions
        points = torch.tensor([[ 1.,  2., 0, 0],
                               [ 4.,  5., 0, 0],
                               [ 7.,  8., 0, 0],
                               [10., 11., 0, 0],
                               [13., 14., 0, 0]], device=device)
        result = pp.voxel_filter(points, [5., 5., 1, 1])
        expect = torch.tensor([[ 2.5000,  3.5000, 0, 0],
                               [ 8.5000,  9.5000, 0, 0],
                               [13.0000, 14.0000, 0, 0]], device=device)
        pp.testing.assert_close(result, expect)


       # test multiple dimensions
        points = torch.tensor([[ 1.,  2.],
                               [ 4.,  5.],
                               [ 7.,  8.],
                               [10., 11.],
                               [13., 14.]], device=device)
        result = pp.voxel_filter(points, [5.])
        expect = torch.tensor([[ 2.5000,  3.5000],
                               [ 8.5000,  9.5000],
                               [13.0000, 14.0000]], device=device)
        pp.testing.assert_close(result, expect)

       # test multiple dimensions
        points = torch.tensor([[ 1.,  2.],
                               [ 4.,  5.],
                               [ 7.,  8.],
                               [10., 11.],
                               [13., 14.]], device=device)
        result1 = pp.voxel_filter(points, [5., 1], random=True)
        assert result1.size(0) == 5, "result1 size is not correct"
        for p in result1:
            assert p in points

        result2 = pp.voxel_filter(points, [5.], random=True)
        assert result2.size(0) == 3
        for p in result1:
            assert p in points


    def test_knn(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # test multiple dimensions
        points = torch.tensor([[0.,  0.,  0.],
                               [1.,  0.,  0.],
                               [0.,  1.,  0.],
                               [0.,  1.,  1.],
                               [10., 1.,  1.],
                               [10., 1., 10.]], device=device)
        result1 = pp.knn_filter(points, k=2, radius=5)
        assert result1.shape == torch.Size([4, 3]), "output shape incorrect"
        result2, mask2 = pp.knn_filter(points, k=2, radius=12, return_mask=True)
        assert result2.shape == torch.Size([5, 3]),  "output shape incorrect"
        assert mask2.sum() == 5, "output mask incorrect"

        result3, mask3 = pp.knn_filter(points, k=2, radius=10, pdim=2, return_mask=True)
        assert result3.shape == torch.Size([6, 3]),  "output shape incorrect"
        assert mask3.sum() == 6, "output mask incorrect"

        result4, mask4 = pp.knn_filter(points, k=2, radius=10, pdim=3, return_mask=True)
        assert result4.shape == torch.Size([5, 3]),  "output shape incorrect"
        assert mask4.sum() == 5, "output mask incorrect"


if __name__=="__main__":
    spline = TestDownsample()
    spline.test_random()
    spline.test_voxel()
    spline.test_knn()
