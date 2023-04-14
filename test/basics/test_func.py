import torch
import pypose as pp


class TestFunc:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_cart_home(self):
        x = torch.randn(2, 3, 3, device=self.device)
        y = pp.homo2cart(pp.cart2homo(x))
        torch.testing.assert_close(x, y), 'conversion wrong!'

    def test_home_cart(self):
        x = torch.randn(2, 3, 3, device=self.device)
        ones = torch.zeros_like(x[...,:1])
        x = torch.cat([x, ones], dim=-1)
        y = pp.homo2cart(x)
        assert torch.all(~torch.isnan(y)), 'values contain Nan!'


if __name__ == '__main__':
    test = TestFunc()
    test.test_cart_home()
    test.test_home_cart()
