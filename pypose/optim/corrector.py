import torch, warnings
from torch import Tensor, nn
from .optimizer import Trivial
from torch.autograd.functional import jacobian


class TrivialScale(nn.Module):
    def __init__(self, kernel, vectorize=True, strategy='reverse-mode'):
        super().__init__()
        self.func = lambda x: self.kernel(x).sum()
        self.vectorize, self.strategy = vectorize, strategy

    def forward(self, E: Tensor, J: Tensor):
        G = jacobian(self.func, E**2, vectorize=self.vectorize, strategy=self.strategy)
        return G * E, J


class SimpleTriggs(nn.Module):
    def __init__(self, kernel, vectorize=True, strategy='reverse-mode'):
        super().__init__()
        self.func = lambda x: self.kernel(x).sum()
        self.vectorize, self.strategy = vectorize, strategy

    def forward(self, E: Tensor, J: Tensor):
        G = jacobian(self.func, E**2, vectorize=self.vectorize, strategy=self.strategy)
        return G.sqrt() * E, G.sqrt() * J
