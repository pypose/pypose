import torch, warnings
from torch import Tensor, nn
from torch.autograd.functional import jacobian


class Trivial(nn.Module):
    r"""
    """
    def __init__(self):
        super().__init__()

    def forward(self, E: Tensor, J: Tensor):
        return E, J


class Scale(nn.Module):
    def __init__(self, kernel, vectorize=True, strategy='forward-mode'):
        super().__init__()
        self.vectorize, self.strategy = vectorize, strategy
        self.kernel = kernel
        self.func = lambda x: self.kernel(x).sum()

    def forward(self, E: Tensor, J: Tensor):
        G = jacobian(self.func, E**2, vectorize=self.vectorize, strategy=self.strategy)
        return G * E, J


class GradScale(nn.Module):
    def __init__(self, kernel, alpha=0.1):
        super().__init__()
        self.alpha = 0.1
        self.func = lambda x: self.kernel(x).sum()

    def forward(self, E: Tensor, J: Tensor):
        S = jacobian(func, E**2, vectorize=True, strategy='reverse-mode').sqrt()
        E = S / (1 - self.alpha) * E
        J = S * (1 - self.alpha * E @ E / E**2) @ J
        return E, J
