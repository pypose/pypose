import torch
from pypose.optim.solver import CG

class TestSolver:
    def sample(self,):
        A = torch.tensor([[0.18029674, 0.31511983, 0.45481117, 0.38600167, 0.28706157],
                          [0.31511983, 1.45753279, 1.55334253, 1.0540756 , 1.07958389],
                          [0.45481117, 1.55334253, 2.36744749, 1.12222781, 1.23653486],
                          [0.38600167, 1.0540756 , 1.12222781, 1.37480584, 1.22232613],
                          [0.28706157, 1.07958389, 1.23653486, 1.22232613, 1.25770048]])
        x = torch.tensor([246.40981904,
                          22.69970642,
                          -56.92392982,
                          -161.7914032,
                          137.26826585])
        b = torch.tensor([[ 2.64306851],
                          [-0.03593633],
                          [ 0.73612658],
                          [ 0.51501254],
                          [-0.26689271]])
        return A, b, x

    def test_cg_fixed(self,):
        A, b, x = self.sample()
        cg = CG()
        torch.testing.assert_close(x, cg(A, b), atol=1e-5, rtol=1e-4)

    def test_cg_batch(self,):
        A, b, x = self.sample()
        cg = CG()
        A_batch = torch.stack([A, A])
        b_batch = torch.stack([b, b])
        x_batch = torch.stack([x, x])
        torch.testing.assert_close(x_batch, cg(A_batch, b_batch), atol=1e-5, rtol=1e-4)

    def test_cg_csr(self,):
        A, b, x = self.sample()
        cg = CG()
        A_csr = A.to_sparse_csr()
        torch.testing.assert_close(x, cg(A_csr, b), atol=1e-5, rtol=1e-4)

if __name__ == '__main__':
    # for dev purpose, will not be executed in CI
    from tqdm import tqdm
    import scipy
    import numpy as np
    for _ in tqdm(range(100)):
        n = 1000
        A = scipy.sparse.random(n, n, density=0.1)
        A = A @ A.T
        b = np.random.randn(n, 1)
        x_gt, info = scipy.sparse.linalg.cg(A, b)

        cg = CG()
        x = cg(torch.from_numpy(A.toarray()), torch.from_numpy(b))
        x = x.cpu().numpy()
        torch.testing.assert_close(x, x_gt, atol=1e-5, rtol=1e-4)
