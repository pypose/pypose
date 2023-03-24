import time
import copy
import torch
import random
import warnings
import torch.linalg
import pypose as pp
from torch import nn
import pypose.optim.solver as ppos
import pypose.optim.kernel as ppok
import pypose.optim.corrector as ppoc
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import pytest

def test_matmul_between_coo_coo():
 
    '''

    s = [ [ 0 0 3 ],
          [ 4 0 5 ] ]
    s * s' =  [ [ 9 15 ] ,
                [ 15 41] ]
    '''

    print()

    i = [[0, 1, 1],
         [2, 0, 2]]
    v = [3, 4, 5]

    test_entries = [
        { 'device': 'cpu' },
        { 'device': 'cuda' },
    ]
    
    
    
    for entry in test_entries:
        print(f'entry = {entry}')
        device = entry['device']

        s = torch.sparse_coo_tensor(i, v, (2, 3), dtype=torch.float32, device=device)
        m = torch.sparse.mm( s, s.transpose(0, 1) )

        answer = torch.tensor([[9, 15], [15, 41]], device = device)

        assert (torch.all(torch.abs(m.to_dense() - answer)) < 1e-9) 

    print('Done')

# def test_matmul_between_bsr_bsr():
#     print()
#     crow_indices = torch.tensor([0, 2, 4])
#     col_indices = torch.tensor([0, 1, 0, 1])
#     values = torch.tensor([[[0, 1, 2], [6, 7, 8]],
#                         [[3, 4, 5], [9, 10, 11]],
#                         [[12, 13, 14], [18, 19, 20]],
#                         [[15, 16, 17], [21, 22, 23]]])

#     test_entries = [
#         { 'device': 'cpu' },
#         { 'device': 'cuda' },
#     ]

#     for entry in test_entries:
#         print(f'entry = {entry}')
#         device = entry['device']

#         bsr = torch.sparse_bsr_tensor(crow_indices, col_indices, values, dtype=torch.float32, device=device)
        
#         with pytest.raises(RuntimeError):
#             torch.sparse.mm( bsr, bsr.transpose(0,1) )
            
def test_SBT():
    i = [[0, 1, 1],[2, 0, 2]]
    v = [3, 4, 5]
    s = torch.sparse_coo_tensor(i, v, (2, 3), dtype=torch.float32, device=device)
    #print(type(s))
    #s= torch.tensor([[9, 15], [15, 41]], device = device)
    #print("start")
    x = pp.SparseBlockTensorNew(s)
    assert ( x is not None)
    #test_matmul_between_coo_coo()


if __name__ == '__main__':
    test_SBT()
