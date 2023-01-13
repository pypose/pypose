import torch
import pypose as pp
from torch import nn
import functorch, functools
from torch.autograd.functional import jacobian


class TestOps:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_pm(self):
        for dtype in [torch.float32, torch.float64, torch.int32]:
            x = pp.pm(torch.tensor([1, 0, -2], dtype=dtype, device=self.device))
            pm = torch.tensor([1, 1, -1], dtype=dtype, device=self.device)
            assert torch.equal(x, pm), "Results incorrect."


if __name__ == '__main__':
    test = TestOps()
    test.test_pm()
