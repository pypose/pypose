import torch, warnings
from torch import Tensor, nn
from torch.autograd import grad
from torch.autograd.functional import jacobian


class TrivialScale(nn.Module):
    '''
    Trivial scale correction.
    '''
    def __init__(self, kernel):
        super().__init__()
        self.func = lambda x: kernel(x).sum()

    def forward(self, E: Tensor, J: Tensor):
        x = E.square().sum(-1, keepdim=True)
        G = jacobian(self.func, x)
        return G * E, J


class SimpleTriggs(nn.Module):
    '''
    Faster yet stable version of Triggs correction
    '''
    def __init__(self, kernel):
        super().__init__()
        self.func = lambda x: kernel(x).sum()

    def forward(self, E: Tensor, J: Tensor):
        x = E.square().sum(-1, keepdim=True)
        s = jacobian(self.func, x).sqrt()
        sj = s.expand_as(E).reshape(-1, 1)
        return s * E, sj * J


class Triggs(nn.Module):
    '''The Triggs correction.
    '''
    def __init__(self, kernel):
        super().__init__()
        self.kernel = kernel

    @torch.enable_grad()
    def compute_grads(self, E):
        x = E.square().sum(-1, keepdim=True).requires_grad_(True)
        y = self.kernel(x).sum()
        g1 = grad(y, x, create_graph=True)[0]
        g2 = grad(g1.sum(), x)[0]
        return x.detach_(), g1.detach_(), g2.detach_()

    def forward(self, E: Tensor, J: Tensor):
        x, g1, g2 = self.compute_grads(E)
        se = g1.sqrt()
        sj = se.expand_as(E).unsqueeze(-1)
        sE, sJ = se * E, sj * J.view(E.shape + (J.shape[-1],))
        M = ~((x==0)|(g2 <=0)).squeeze(-1)
        alpha = 1 - (1 + 2*x[M]*g2[M]/g1[M]).clamp(min=0).sqrt()
        sE[M] = se[M] / (1 - alpha)
        Q = torch.einsum('...d,...k,...kl->...dl', E[M], E[M], sJ[M])
        sJ[M] = sJ[M] - (alpha / x[M]).unsqueeze(-1) * Q
        return sE, sJ.view_as(J)
