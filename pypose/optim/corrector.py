import torch, warnings
from torch import Tensor, nn
from torch.autograd import grad
from torch.autograd.functional import jacobian


class GradScale(nn.Module):
    r'''
    The gradient scale correction of model residual and Jacobian for the non-linear least
    squares problems.

    .. math::
        \begin{align*}
            \mathbf{E}_i &= \frac{\mathrm{d} \rho(c_i)}{\mathrm{d} c_i} \mathbf{E}_i\\
            \mathbf{J}_i &= \frac{\mathrm{d} \rho(c_i)}{\mathrm{d} c_i} \mathbf{J}_i
        \end{align*},

    where :math:`\mathbf{E}_i` and :math:`\mathbf{J}_i` is the :math:`i`-th item of the model
    residual and Jacobian, respectively. :math:`\rho()` is the kernel function and
    :math:`c_i = \mathbf{E}_i^T\mathbf{E}_i` is the point to compute the gradient.

    Args:
        kernel (nn.Module): the robust kernel (cost) function.
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
            The :obj:`.forward()` function is not supposed to be directly called by PyPose users.
            It will be called internally by optimizers such as :meth:`pypose.optim.GaussNewton` and
            :meth:`pypose.optim.LevenbergMarquardt`.
        '''
        x = E.square().sum(-1, keepdim=True)
        g = jacobian(self.func, x)
        s = s.expand_as(E).reshape(-1, 1)
        return g * E, s * J


class FastTriggs(nn.Module):
    r'''
    Faster yet stable version of Triggs correction of model residual and Jacobian for the
    non-linear least squares problems. It removes the 2nd order derivative in the full Triggs
    correction, which leads a faster and numerically stable solution. This is because most kernel
    functions have a negative Hessian, which can lead a 2nd order optimizer unstable.

    .. math::
        \begin{align*}
            \mathbf{E}_i &= \sqrt{\rho'(c_i)} \mathbf{E}_i\\
            \mathbf{J}_i &= \sqrt{\rho'(c_i)} \mathbf{J}_i
        \end{align*},
    
    where :math:`\mathbf{E}_i` and :math:`\mathbf{J}_i` is the :math:`i`-th item of the model
    residual and Jacobian, respectively. :math:`\rho()` is the kernel function and
    :math:`c_i = \mathbf{E}_i^T\mathbf{E}_i` is the point to compute the gradient.
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
            The :obj:`.forward()` function is not supposed to be directly called by PyPose users.
            It will be called internally by optimizers such as :meth:`pypose.optim.GaussNewton` and
            :meth:`pypose.optim.LevenbergMarquardt`.
        '''
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
        r'''
        Args:
            E (Tensor): the model residual.
            J (Tensor): the model Jacobian.

        Returns:
            tuple of Tensors: the corrected model residual and model Jacobian.

        Note:
            The :obj:`.forward()` function is not supposed to be directly called by PyPose users.
            It will be called internally by optimizers such as :meth:`pypose.optim.GaussNewton` and
            :meth:`pypose.optim.LevenbergMarquardt`.
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
