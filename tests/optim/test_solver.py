import time
import torch
import pypose as pp

import torch

from pypose.optim.solver import CG

from tqdm import tqdm
for _ in tqdm(range(1000)):
    import scipy
    import numpy as np
    # example linear solver
    A = scipy.sparse.random(1000, 1000, density=0.01)
    A = A @ A.T
    b = np.random.randn(1000, 1)
    x_gt, info = scipy.sparse.linalg.cg(A, b)

    # example pp solver
    cg = CG()
    x = cg(torch.from_numpy(A.toarray()), torch.from_numpy(b))
    x = x.cpu().numpy()
    torch.testing.assert_close(x, x_gt, atol=1e-5, rtol=1e-4)
