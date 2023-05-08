import torch
from pypose import hasnan


class TestChecking:
    def test_hasnan(self):

        L1 = [[1, 3], [4, [5, 6]], 7, [8, torch.tensor([0, -1.0999])]]
        L2 = [[1, 3], [4, [torch.tensor(float('nan')), 6]], 7, [8, torch.tensor([0, 1.])]]
        L3 = [[1, 3], [4, [5, 6]], torch.tensor(float('nan')), [8, torch.tensor([0, 1.])]]
        L4 = [[1, 3], [4, [5, 6]], 7, [8, torch.tensor([float('nan'), -1.0999])]]
        L5 = [[torch.tensor([float('nan'), -1.0999]), 3], [4, [5, 6]], 7, [8, 9]]
        L6 = [[torch.tensor([1, -1.0999]), 3], [4, [float('nan'), 6]], 7, [8, 9]]

        assert hasnan(L1) is False
        assert hasnan(L2) is True
        assert hasnan(L3) is True
        assert hasnan(L4) is True
        assert hasnan(L5) is True
        assert hasnan(L6) is True


if __name__ == '__main__':
    test = TestChecking()
    test.test_hasnan()
