import torch
from torch import nn


class SO3(torch.Tensor):
    group_name = 'SO3'
    group_id = 1
    manifold_dim = 3
    embedded_dim = 4

    def __new__(cls, data=None, requires_grad=True):
        if data is None:
            data = torch.empty(0)
        return torch.Tensor._make_subclass(cls, data, requires_grad)

    def Exp(self):
        return self.exp() + self.sum()
    
    def __repr__(self):
        return self.__class__.__name__ + ':\n' + super().__repr__()
    # __torch_function__ = torch._C._disabled_torch_function_impl


if __name__ == '__main__':
    x = SO3(torch.randn(8, 3), requires_grad=True)
    print(x.sum())
    print(x.Exp())
    y = x.sum().Exp()
    print(y)
    y.backward()
    print(x.grad)
    z = x[:,1] + x[:,2]
    z.add(1).Exp().sum().backward()
    print(x.grad)
