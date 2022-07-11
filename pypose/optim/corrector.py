import torch, warnings
from torch import Tensor, nn
from torch.autograd import grad
from torch.autograd.functional import jacobian


class FastTriggs(nn.Module):
    r'''
    Faster yet stable version of Triggs correction of model residual and Jacobian.

    .. math::
        \begin{align*}
            \mathbf{E}_i^\rho &= \sqrt{\rho'(c_i)} \mathbf{E}_i\\
            \mathbf{J}_i^\rho &= \sqrt{\rho'(c_i)} \mathbf{J}_i
        \end{align*},
    
    where :math:`\mathbf{E}_i` and :math:`\mathbf{J}_i` is the :math:`i`-th item of the model
    residual and Jacobian, respectively. :math:`\rho()` is the kernel function and
    :math:`c_i = \mathbf{E}_i^T\mathbf{E}_i` is the point to compute the gradient.

    Args:
        kernel (nn.Module): the robust kernel (cost) function.

    Note:
        This implementation has a faster and numerically stable solution than :meth:`Triggs`
        due to the removal of 2nd order derivatives, since most kernel functions have negative
        Hessians, which can lead a 2nd order optimizer unstable. It basically aims to solve

        .. math::
            \bm{\theta}^* = \arg\min_{\bm{\theta}} \mathbf{g}(\bm{x})
                        = \arg\min_{\bm{\theta}} \sum_i \rho(\mathbf{E}_i^T \mathbf{E}_i),

        where :math:`\mathbf{E}_i = \bm{y}_i-\bm{f}(\bm{\theta},\bm{x}_i)` and
        :math:`\bm{f}(\bm{\theta}, \bm{x})` is the model, :math:`\bm{\theta}` is the parameters
        to be optimized, :math:`\bm{x}` is the model inputs, :math:`\bm{y}` is the model targets.
        Considering the 1st order Taylor expansion of the model
        :math:`\bm{f}(\bm{\theta}+\delta) \approx \bm{f}(\bm{\theta}) + \mathbf{J}_i \bm{\theta}`.
        If we take :math:`c_i = \mathbf{E}_i^T \mathbf{E}_i` and set the first derivative of
        :math:`\mathbf{g}(\bm{\delta})` to zero, we have

        .. math::
            \frac{\partial \bm{g}}{\partial \bm{\delta}} 
            = \sum_i \frac{\partial \rho}{\partial c_i} \frac{\partial c_i}{\partial \bm{\delta}}
            = \bm{0}

        This leads to

        .. math::
            \sum_i \frac{\partial \rho}{\partial c_i} \mathbf{J}_i^T \mathbf{J}_i \bm{\delta}
            = - \sum_i \frac{\partial \rho}{\partial c_i} \mathbf{J}_i^T \mathbf{E}_i

        Rearrange the gradient of :math:`\rho`, we have

        .. math::
            \sum_i \left(\sqrt{\frac{\partial \rho}{\partial c_i}} \mathbf{J}_i\right)^T 
                \left(\sqrt{\frac{\partial \rho}{\partial c_i}} \mathbf{J}_i\right) \bm{\delta}
            = - \sum_i \left(\sqrt{\frac{\partial \rho}{\partial c_i}} \mathbf{J}_i\right)^T 
                \left(\sqrt{\frac{\partial \rho}{\partial c_i}} \mathbf{E}_i\right)

        This gives us the corrected model residual :math:`\mathbf{E}_i^\rho` and Jacobian
        :math:`\mathbf{J}_i^\rho`, which is the solution to the standard 2nd order optimizers
        such as :meth:`pypose.optim.GN` and :meth:`pypose.optim.LM`.

        .. math::
            \sum_i {\mathbf{J}_i^\rho}^T \mathbf{J}_i^\rho \bm{\delta}
            = - \sum_i {\mathbf{J}_i^\rho}^T \mathbf{E}_i^\rho
    '''
    def __init__(self, kernel):
        super().__init__()
        self.func = lambda x: kernel(x).sum()

    def forward(self, E: Tensor, J: Tensor):
        r'''
        Args:
            E (Tensor): the model residual.
            J (Tensor): the model Jacobian.

        Returns:
            tuple of Tensors: the corrected model residual and model Jacobian.

        Note:
            The :obj:`.forward()` function is not supposed to be directly called by PyPose
            users. It will be called internally by optimizers such as
            :meth:`pypose.optim.GN` and :meth:`pypose.optim.LM`.
        '''
        x = E.square().sum(-1, keepdim=True)
        s = jacobian(self.func, x).sqrt()
        sj = s.expand_as(E).reshape(-1, 1)
        return s * E, sj * J


class Triggs(nn.Module):
    '''The Triggs correction correction of model residual and Jacobian.
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
        r'''
        Args:
            E (Tensor): the model residual.
            J (Tensor): the model Jacobian.

        Returns:
            tuple of Tensors: the corrected model residual and model Jacobian.

        Note:
            The :obj:`.forward()` function is not supposed to be directly called by PyPose
            users. It will be called internally by optimizers such as :meth:`pypose.optim.GN`
            and :meth:`pypose.optim.LM`.
        '''
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
