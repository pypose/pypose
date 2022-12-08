
import pytest

import torch

def test_matmul_between_coo_coo():
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

def test_matmul_between_bsr_bsr():
    print()
    crow_indices = torch.tensor([0, 2, 4])
    col_indices = torch.tensor([0, 1, 0, 1])
    values = torch.tensor([[[0, 1, 2], [6, 7, 8]],
                        [[3, 4, 5], [9, 10, 11]],
                        [[12, 13, 14], [18, 19, 20]],
                        [[15, 16, 17], [21, 22, 23]]])

    test_entries = [
        { 'device': 'cpu' },
        { 'device': 'cuda' },
    ]

    for entry in test_entries:
        print(f'entry = {entry}')
        device = entry['device']

        bsr = torch.sparse_bsr_tensor(crow_indices, col_indices, values, dtype=torch.float32, device=device)
        
        with pytest.raises(RuntimeError):
            torch.sparse.mm( bsr, bsr.transpose(0,1) )

