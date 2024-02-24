import torch
import pypose as pp


class TestOps:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_pm(self):
        for dtype in [torch.float32, torch.float64, torch.int32]:
            x = pp.pm(torch.tensor([1, 0, -2], dtype=dtype, device=self.device))
            pm = torch.tensor([1, 1, -1], dtype=dtype, device=self.device)
            assert torch.equal(x, pm), "Results incorrect."

    def test_cum(self):
        N = 4
        x = pp.randn_SO3(N, dtype=torch.float64)

        O1 = x[3] @ x[2] @ x[1] @ x[0]
        O2 = pp.cumprod(x, dim=0, left=True)[N-1]
        pp.testing.assert_close(O1, O2, rtol=1e-4, atol=1e-4)

        O3 = x[0] @ x[1] @ x[2] @ x[3]
        O4 = pp.cumprod(x, dim=0, left=False)[N-1]
        pp.testing.assert_close(O3, O4, rtol=1e-4, atol=1e-4)

        O5 = x[3] @ x[2] @ x[1] @ x[0]
        O6 = pp.cummul(x, dim=0, left=True)[N-1]
        pp.testing.assert_close(O5, O6, rtol=1e-4, atol=1e-4)

        O7 = x[0] @ x[1] @ x[2] @ x[3]
        O8 = pp.cummul(x, dim=0, left=False)[N-1]
        pp.testing.assert_close(O7, O8, rtol=1e-4, atol=1e-4)

        y = x.clone()
        pp.cumprod_(y, dim=0, left=True)
        pp.testing.assert_close(O1, y[N-1], rtol=1e-4, atol=1e-4)

        z = x.clone()
        pp.cumprod_(z, dim=0, left=False)
        pp.testing.assert_close(O3, z[N-1], rtol=1e-4, atol=1e-4)

        y = x.clone()
        pp.cummul_(y, dim=0, left=True)
        pp.testing.assert_close(O1, y[N-1], rtol=1e-4, atol=1e-4)

        z = x.clone()
        pp.cummul_(z, dim=0, left=False)
        pp.testing.assert_close(O3, z[N-1], rtol=1e-4, atol=1e-4)

        left = lambda a, b: b @ a
        O9 = pp.cumops(x, dim=0, ops=left)[N-1]
        pp.testing.assert_close(O1, O9, rtol=1e-4, atol=1e-4)

        right = lambda a, b: a @ b
        OX = pp.cumops(x, dim=0, ops=right)[N-1]
        pp.testing.assert_close(O3, OX, rtol=1e-4, atol=1e-4)

if __name__ == '__main__':
    test = TestOps()
    test.test_pm()
    test.test_cum()
