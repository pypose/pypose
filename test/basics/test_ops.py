import torch
import numpy as np
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

    def test_msqrt(self):
        matrixSize = 5
        A = np.random.rand(matrixSize, matrixSize)
        B = np.dot(A, A.transpose())
        matrix = torch.tensor(B, dtype=torch.float).unsqueeze(0)
        out = pp.msqrt(matrix)
        assert torch.all(torch.bmm(out, out) - matrix < 1e-1), 'msqrt error too large'


if __name__ == '__main__':
    test = TestOps()
    test.test_pm()
    test.test_msqrt()
