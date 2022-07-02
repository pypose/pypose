import torch, sys, math, warnings
from ..functional import modjac
from ...lietensor import LieTensor
from torch.optim import Optimizer
from .gauss_newton import GaussNewton
import torch.nn.functional as F


class LevenbergMarquardt(GaussNewton):
    r'''
    The `Levenberg-Marquardt (LM) algorithm
    <https://en.wikipedia.org/wiki/Levenberg-Marquardt_algorithm>`_, which is also known as the
    damped least-squares (DLS) method for solving non-linear least squares problems. This
    implementation is for optimizing the model parameters to approximate the targets, which can
    be a Tensor/LieTensor or a tuple of Tensors/LieTensors.

    .. math::
        \bm{\theta}^* = \arg\min_{\bm{\theta}}\sum_i \|\bm{y}_i - \bm{f}(\bm{\theta}, \bm{x}_i)\|^2,

    where :math:`\bm{f}(\bm{\theta}, \bm{x})` is the model, :math:`\bm{\theta}` is the parameters
    to be optimized, and :math:`\bm{x}` is the model inputs.

    .. math::
       \begin{aligned}
            &\rule{113mm}{0.4pt}                                                                 \\
            &\textbf{input}: \lambda \geq 0~\text{(damping)}, \bm{\theta}_0~\text{(params)},
                \bm{f}~\text{(model)}, \bm{x}~(\text{inputs}), \bm{y}~(\text{targets})           \\
            &\rule{113mm}{0.4pt}                                                                 \\
            &\textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}                         \\
            &\hspace{5mm} \mathbf{J} \leftarrow {\dfrac {\partial \bm{f}}
                {\partial \bm{\theta}_{t-1}}}                                                    \\
            &\hspace{5mm} \mathbf{A} \leftarrow \mathbf{J}^T \mathbf{J} 
                       + \lambda \mathrm{diag}(\mathbf{J}^T \mathbf{J}).\mathrm{clamp(min, max)} \\
            &\hspace{5mm} \mathbf{E} = \bm{y} - \bm{f(\bm{\theta}_{t-1}, \bm{x})}                \\
            &\hspace{5mm} \mathbf{L} = \mathrm{cholesky\_decomposition}(\mathbf{A})              \\
            &\hspace{5mm} \bm{\delta}=\mathrm{cholesky\_solve}(\mathbf{J}^T \mathbf{E}, \bm{L})  \\
            &\hspace{5mm} \bm{\theta}_t \leftarrow \bm{\theta}_{t-1} + \bm{\delta}               \\
            &\rule{113mm}{0.4pt}                                                          \\[-1.ex]
            &\bf{return} \:  \theta_t                                                     \\[-1.ex]
            &\rule{113mm}{0.4pt}                                                          \\[-1.ex]
       \end{aligned}

    Args:
        model (nn.Module): a module containing learnable parameters.
        damping (float): Levenberg's damping factor (positive number).
        min (float, optional): the lower-bound of the matrix diagonal to inverse.
        max (float, optional): the upper-bound of the matrix diagonal to inverse.
    '''
    def __init__(self, model, damping, min=1e-6, max=1e32):
        self.model = model
        assert damping > 0, ValueError("Invalid damping factor: {}".format(damping))
        defaults = dict(damping=damping, min=min, max=max)
        Optimizer.__init__(self, params=model.parameters(), defaults=defaults)

    @torch.no_grad()
    def step(self, inputs, targets=None):
        r'''
        Performs a single optimization step.

        Args:
            inputs (Tensor/LieTensor or tuple of Tensors/LieTensors): the inputs to the model.
            targets (Tensor/LieTensor or tuple of Tensors/LieTensors): the model targets to optimize.
                If not given, the squared model outputs are minimized. Defaults: ``None``.

        Return:
            Tensor: the minimized model error, i.e., :math:`\|\bm{y} - \bm{f}(\bm{\theta}, \bm{x})\|^2`.

        Note:
            The (non-negative) damping factor :math:`\lambda` can be adjusted at each iteration. If
            reduction of the residual is rapid, a smaller value can be used, bringing the algorithm
            closer to the Gauss-Newton algorithm, whereas if an iteration gives insufficient reduction
            in the residual, :math:`\lambda` can be increased, giving a step closer to the gradient
            descent direction.

        Note:
            Different from PyTorch optimizers like
            `SGD <https://pytorch.org/docs/stable/generated/torch.optim.SGD.html>`_, where the model
            error has to be a scalar, the model output of :obj:`LM` can be a Tensor/LieTensor or a
            tuple of Tensors/LieTensors.

        Example:
            Optimizing a simple module to **approximate pose inversion**.

            >>> class PoseInv(nn.Module):
            ...     def __init__(self, *dim):
            ...         super().__init__()
            ...         self.pose = pp.Parameter(pp.randn_se3(*dim))
            ...
            ...     def forward(self, inputs):
            ...         return (self.pose.Exp() @ inputs).Log()
            ...
            >>> posinv = PoseInv(2, 2)
            >>> inputs = pp.randn_SE3(2, 2)
            >>> optimizer = pp.optim.LM(posinv, damping=1e-6)
            ...
            >>> for idx in range(10):
            ...     loss = optimizer.step(inputs)
            ...     print('Pose Inversion loss %.7f @ %d it'%(loss, idx))
            ...     if loss < 1e-5:
            ...         print('Early Stoping with loss:', loss.item())
            ...         break
            ...
            Pose Inversion error: 1.6600330 @ 0 it
            Pose Inversion error: 0.1296970 @ 1 it
            Pose Inversion error: 0.0008593 @ 2 it
            Pose Inversion error: 0.0000004 @ 3 it
            Early Stoping with error: 4.443569991963159e-07
        '''
        E = self._residual(inputs, targets)
        for pg in self.param_groups:
            numels = [p.numel() for p in pg['params'] if p.requires_grad]
            J = modjac(self.model, inputs, flatten=True)
            A = J.T @ J
            A.diagonal().add_(pg['damping'] * A.diagonal().clamp(pg['min'], pg['max']))
            D = (J.T @ E).cholesky_solve(torch.linalg.cholesky(A))
            D = torch.split(D, numels)
            [p.add_(d.view(p.shape)) for p, d in zip(pg['params'], D) if p.requires_grad]
        return self._residual(inputs, targets).norm()
