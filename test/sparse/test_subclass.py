
import pytest

import torch

class MyTensor(torch.Tensor):
    def __init__(self, **data):
        s = torch.sparse_coo_tensor(**data)

        # copy from s to self

    @staticmethod
    def __new__(cls, indices, values, shape):
        s = torch.sparse_coo_tensor(indices, values, shape)


    # @staticmethod
    # def __new__(cls, *data):
    #     sbt = data[0] if isinstance(data[0], torch.Tensor) else torch.Tensor(*data)
    #     # return torch.Tensor.as_subclass(sbt, MyTensor)
    #     print(f'cls = {cls}')
    #     return super().__new__(cls, *data)

# def test_creation():
#     t = torch.rand((3,3))
#     my_tensor = MyTensor( t )

#     print(f'type(my_tensor) = {type(my_tensor)}')

# def test_sparse_creation():
#     i = [[0, 1, 1],[2, 0, 2]]
#     v = [3, 4, 5]
#     s = torch.sparse_coo_tensor(i, v, (2, 3), dtype=torch.float32, device='cpu')

#     my_tensor = MyTensor( s )

#     print(f'type(my_tensor) = {type(my_tensor)}')

# def test_init():
#     i = [[0, 1, 1],[2, 0, 2]]
#     v = [3, 4, 5]
#     s = torch.sparse_coo_tensor(i, v, (2, 3))

#     my_tensor = MyTensor(i, v, (2, 3))
#     print(f'type(my_tensor) = {type(my_tensor)}')
