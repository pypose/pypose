import torch, sys, math, warnings
from .jacobian import modjac
from ..lietensor import LieTensor
from torch.optim import Optimizer


class LM(Optimizer):
    r'''
    The `Levenberg-Marquardt (LM) algorithm
    <https://en.wikipedia.org/wiki/Levenberg-Marquardt_algorithm>`_, which isalso known as the
    damped least-squares (DLS) method for solving non-linear least squares problems. This
    implementation is for solving the model parameters to minimize (or maximize) the model output,
    which can be a Tensor/LieTensor or a tuple of Tensors/LieTensors.

    .. math::
       \begin{aligned}
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{input}: \lambda \geq 0~\text{(dampening)}, \bm{\theta}_0 \text{ (params)},
                \bm{f}(\bm{\theta}) \text{ (model)}, \text{maximize}                             \\
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}                         \\
            &\hspace{5mm} \mathbf{J} \leftarrow {\dfrac {\partial \bm{f}}{\partial \bm{\theta}}} \\
            &\hspace{5mm} \mathbf{A} \leftarrow \mathbf{J}^T \mathbf{J}  + \lambda \mathbf{I}    \\
            &\hspace{5mm} \textbf{try}                                                           \\
            &\hspace{10mm} \mathbf{L} = \mathrm{cholesky\_decomposition}(\mathbf{A})             \\
            &\hspace{10mm} \bm{\delta} = \mathrm{cholesky\_solve}(\mathbf{J}^T, \bm{L})          \\
            &\hspace{5mm} \textbf{except}                                                        \\
            &\hspace{10mm} \bm{\delta} = \mathrm{pseudo\_inverse}(\mathbf{A}) \mathbf{J}^T       \\
            &\hspace{5mm} M = 1 \: \textbf{if}~\text{maximize} \: \textbf{else} -1               \\
            &\hspace{5mm} \bm{\theta}_t \leftarrow \bm{\theta}_{t-1} + M \bm{\delta} \mathbf{E}_t\\
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
            &\bf{return} \:  \theta_t                                                     \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
       \end{aligned}

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        dampening (float): Levenberg's dampening factor (non-negative number) to prevent
            singularity.
        maximize (bool, optional): maximize the params based on the objective, instead of
            minimizing (default: False)

    Note:
        The (non-negative) damping factor :math:`\lambda` can be adjusted at each iteration. If
        reduction of the residual is rapid, a smaller value can be used, bringing the algorithm
        closer to the Gauss-Newton algorithm, whereas if an iteration gives insufficient reduction
        in the residual, :math:`\lambda` can be increased, giving a step closer to the gradient
        descent direction.

    Note:
        Different from PyTorch optimizers like
        `SGD <https://pytorch.org/docs/stable/generated/torch.optim.SGD.html>`_, where the model
        loss has to be a scalar, the model output of :obj:`LM` can be a Tensor/LieTensor or a
        tuple of Tensors/LieTensors.

    Example:
        Optimizing a simple module to **approximate pose inversion**.

        >>> class PoseInv(nn.Module):
        ...     def __init__(self, *dim):
        ...         super().__init__()
        ...         self.pose = pp.Parameter(pp.randn_se3(*dim))
        ...
        ...     def forward(self, inputs):
        ...         return (self.pose.Exp() @ inputs).Log().abs()
        ...
        >>> posnet, pose = PoseInv(2, 2), pp.randn_SE3(2, 2)
        >>> optimizer = pp.optim.LM(posnet, dampening=1e-6)
        ...
        >>> for idx in range(10):
        ...     loss = optimizer.step(pose)
        ...     error = loss.mean()
        ...     print('Pose Inversion error %.7f @ %d it'%(error, idx))
        ...     if error < 1e-5:
        ...         print('Early Stoping with error:', error.item())
        ...         break
        ...
        Pose Inversion error 1.1270601 @ 0 it
        Pose Inversion error 0.2298058 @ 1 it
        Pose Inversion error 0.0203174 @ 2 it
        Pose Inversion error 0.0001056 @ 3 it
        Pose Inversion error 0.0000001 @ 4 it
        Early Stoping with error: 7.761021691976566e-08
    '''
    def __init__(self, model, dampening, maximize=False):
        self.model = model
        assert dampening >= 0, ValueError("Invalid dampening value: {}".format(dampening))
        defaults = dict(dampening=dampening, maximize=maximize)
        super().__init__(model.parameters(), defaults)

    @torch.no_grad()
    def step(self, inputs=None):
        r'''
        Performs a single optimization step.

        Args:
            inputs (Tensor/LieTensor or tuple of Tensors/LieTensors): inputs to the model. Defaults
                to ``None``. Cannot be ``None`` if the model requires inputs.

        Return:
            Tensor/LieTensor or tuple of Tensors/LieTensors: the minimized (maximized) model output
            that is taken as a loss or an objective.
        '''
        loss = self.model(inputs)
        if isinstance(loss, tuple):
            L = torch.cat([l.tensor().view(-1, 1) if isinstance(l, LieTensor) \
                            else l.view(-1, 1) for l in loss])
        else:
            L = loss.tensor().view(-1, 1) if isinstance(loss, LieTensor) else loss.view(-1, 1)
        for group in self.param_groups:
            numels = [p.numel() for p in group['params'] if p.requires_grad]
            J = modjac(self.model, inputs, flatten=True)
            A = (J.T @ J) + group['dampening'] * torch.eye(J.size(-1)).to(J)
            try: # Faster but sometimes singular error
                D = J.T.cholesky_solve(torch.linalg.cholesky(A))
            except: # Slower but singular is fine
                D = (A.pinverse() @ J.T)
                warnings.warn("Using pseudo inverse due to singular matrix.", UserWarning)
            D = torch.split(D, numels)
            maximize = 1 if group['maximize'] else -1
            [p.add_(maximize * (d @ L).view(p.shape)) \
                for p, d in zip(group['params'], D) if p.requires_grad]
        return loss
