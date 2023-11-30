import torch
import pypose as pp
crow_indices = torch.tensor([0, 2, 4])
col_indices = torch.tensor([0, 1, 0, 1])
values = torch.tensor([1, 2, 3, 4])
csr = torch.sparse_csr_tensor(crow_indices, col_indices, values, dtype=torch.float64)
csr
csr.to_dense()

crow_indices = torch.tensor([0, 2, 4])
col_indices = torch.tensor([0, 1, 0, 1])
values = torch.tensor([[[0, 1, 2], [6, 7, 8]],
                       [[3, 4, 5], [9, 10, 11]],
                       [[12, 13, 14], [18, 19, 20]],
                       [[15, 16, 17], [21, 22, 23]]])
bsr = torch.sparse_bsr_tensor(crow_indices, col_indices, values, dtype=torch.float64)
print(torch.matmul(bsr, bsr.mT))
