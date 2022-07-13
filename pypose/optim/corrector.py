import torch, warnings
from torch import Tensor, nn
from torch.autograd import grad
from torch.autograd.functional import jacobian


class FastTriggs(nn.Module):
    r'''
    Faster yet stable version of Triggs correction of model residual and Jacobian.

    .. math::
        \begin{align*}
            \mathbf{R}_i^\rho &= \sqrt{\rho'(c_i)} \mathbf{R}_i\\
            \mathbf{J}_i^\rho &= \sqrt{\rho'(c_i)} \mathbf{J}_i
        \end{align*},
    
    where :math:`\mathbf{R}_i` and :math:`\mathbf{J}_i` are the :math:`i`-th item of the
    model residual and Jacobian, respectively. :math:`\rho()` is the kernel function and
    :math:`c_i = \mathbf{R}_i^T\mathbf{R}_i` is the point to compute the gradient.

    Args:
        kernel (nn.Module): the robust kernel (cost) function.

    Note:
        This implementation has a faster and numerically stable solution than :meth:`Triggs`.
        It removes the kernel's 2nd order derivatives (often negative), which can lead a 2nd
        order optimizer unstable. It basically aims to solve

        .. math::
            \bm{\theta}^* = \arg\min_{\bm{\theta}} \mathbf{g}(\bm{x})
                        = \arg\min_{\bm{\theta}} \sum_i \rho(\mathbf{R}_i^T \mathbf{R}_i),

        where :math:`\mathbf{R}_i = \bm{f}(\bm{\theta},\bm{x}_i) - \bm{y}_i` and
        :math:`\bm{f}(\bm{\theta}, \bm{x})` is the model, :math:`\bm{\theta}` is the parameters
        to be optimized, :math:`\bm{x}` is the model inputs, :math:`\bm{y}` is the model targets.
        Considering the 1st order Taylor expansion of the model
        :math:`\bm{f}(\bm{\theta}+\bm{\delta})\approx\bm{f}(\bm{\theta})+\mathbf{J}_i\bm{\delta}`.
        If we take :math:`c_i = \mathbf{R}_i^T \mathbf{R}_i` and set the first derivative of
        :math:`\mathbf{g}(\bm{\delta})` to zero, we have

        .. math::
            \frac{\partial \bm{g}}{\partial \bm{\delta}} 
            = \sum_i \frac{\partial \rho}{\partial c_i} \frac{\partial c_i}{\partial \bm{\delta}}
            = \bm{0}

        This leads to

        .. math::
            \sum_i \frac{\partial \rho}{\partial c_i} \mathbf{J}_i^T \mathbf{J}_i \bm{\delta}
            = - \sum_i \frac{\partial \rho}{\partial c_i} \mathbf{J}_i^T \mathbf{R}_i

        Rearrange the gradient of :math:`\rho`, we have

        .. math::
            \sum_i \left(\sqrt{\frac{\partial \rho}{\partial c_i}} \mathbf{J}_i\right)^T 
                \left(\sqrt{\frac{\partial \rho}{\partial c_i}} \mathbf{J}_i\right) \bm{\delta}
            = - \sum_i \left(\sqrt{\frac{\partial \rho}{\partial c_i}} \mathbf{J}_i\right)^T 
                \left(\sqrt{\frac{\partial \rho}{\partial c_i}} \mathbf{R}_i\right)

        This gives us the corrected model residual :math:`\mathbf{R}_i^\rho` and Jacobian
        :math:`\mathbf{J}_i^\rho`, which ends with the same problem formulation as the
        standard 2nd order optimizers such as :meth:`pypose.optim.GN` and
        :meth:`pypose.optim.LM`.

        .. math::
            \sum_i {\mathbf{J}_i^\rho}^T \mathbf{J}_i^\rho \bm{\delta}
            = - \sum_i {\mathbf{J}_i^\rho}^T \mathbf{R}_i^\rho
    '''
    def __init__(self, kernel):
        super().__init__()
        self.func = lambda x: kernel(x).sum()

    def forward(self, R: Tensor, J: Tensor):
        r'''
        Args:
            R (Tensor): the model residual.
            J (Tensor): the model Jacobian.

        Returns:
            tuple of Tensors: the corrected model residual and model Jacobian.

        Note:
            The users basically only need to call the constructor, while the :obj:`.forward()`
            function is not supposed to be directly called by PyPose users. It will be called
            internally by optimizers such as :meth:`pypose.optim.GN` and :meth:`pypose.optim.LM`.
        '''
        x = R.square().sum(-1, keepdim=True)
        s = jacobian(self.func, x).sqrt()
        sj = s.expand_as(R).reshape(-1, 1)
        return s * R, sj * J


class Triggs(nn.Module):
    r'''The Triggs correction of model residual and Jacobian.

    .. math::
        \begin{align*}
            \mathbf{R}_i^\rho &= \frac{\sqrt{\rho'(c_i)}}{1 - \alpha} \mathbf{R}_i,\\
            \mathbf{J}_i^\rho &= \sqrt{\rho'(c_i)} \left(\mathbf{I} - \alpha
                \frac{\mathbf{R}_i^T\mathbf{R}_i}{\|\mathbf{R}_i\|^2} \right) \mathbf{J}_i,
        \end{align*}
    
    where :math:`\alpha` is a root of

    .. math::
        \frac{1}{2} \alpha^2 - \alpha - \frac{\rho''}{\rho'} \|\mathbf{R}_i\|^2 = 0.

    :math:`\mathbf{R}_i` and :math:`\mathbf{J}_i` are the :math:`i`-th item of the model
    residual and Jacobian, respectively. :math:`\rho()` is the kernel function and
    :math:`c_i = \mathbf{R}_i^T\mathbf{R}_i` is the evaluation point.

    Args:
        kernel (nn.Module): the robust kernel (cost) function.

    Note:
        This implementation thanks to Eq. (11) of the following paper.

        * Bill Triggs, etc., `Bundle Adjustment -- A Modern Synthesis
          <https://link.springer.com/chapter/10.1007/3-540-44480-7_21>`_, International
          Workshop on Vision Algorithms, 1999.
    
    Warning:

        The :meth:`FastTriggs` corrector is preferred when the kernel function has a
        negative 2nd order derivative.
    '''
    def __init__(self, kernel):
        super().__init__()
        self.kernel = kernel

    @torch.enable_grad()
    def compute_grads(self, R):
        x = R.square().sum(-1, keepdim=True).requires_grad_(True)
        y = self.kernel(x).sum()
        g1 = grad(y, x, create_graph=True)[0]
        g2 = grad(g1.sum(), x)[0]
        return x.detach_(), g1.detach_(), g2.detach_()

    def forward(self, R: Tensor, J: Tensor):
        r'''
        Args:
            R (Tensor): the model residual.
            J (Tensor): the model Jacobian.

        Returns:
            tuple of Tensors: the corrected model residual and model Jacobian.

        Note:
            The users basically only need to call the constructor, while the :obj:`.forward()`
            function is not supposed to be directly called by PyPose users. It will be called
            internally by optimizers such as :meth:`pypose.optim.GN` and :meth:`pypose.optim.LM`.
        '''
        x, g1, g2 = self.compute_grads(R)
        se = g1.sqrt()
        sj = se.expand_as(R).unsqueeze(-1)
        sR, sJ = se * R, sj * J.view(R.shape + (J.shape[-1],))
        M = ~((x==0)|(g2 <=0)).squeeze(-1)
        alpha = 1 - (1 + 2*x[M]*g2[M]/g1[M]).clamp(min=0).sqrt()
        sR[M] = se[M] / (1 - alpha)
        Q = torch.einsum('...d,...k,...kl->...dl', R[M], R[M], sJ[M])
        sJ[M] = sJ[M] - (alpha / x[M]).unsqueeze(-1) * Q
        return sR, sJ.view_as(J)
