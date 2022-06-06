import torch, sys, math, warnings
from .jacobian import modjac
from torch.optim import Optimizer
import torch.autograd.functional as F


class LM(Optimizer):
    r'''
    The `Levenberg-Marquardt (LM) algorithm
    <https://en.wikipedia.org/wiki/Levenberg-Marquardt_algorithm>`_, which also known as the damped
    least-squares (DLS) method for solving non-linear least squares problems.

    .. math::
       \begin{aligned}
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{input}: \lambda > 0~\text{(dampening)}, \bm{\theta}_0 \text{ (params)},
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
        model (torch.nn.Module): a PyTorch model that takes Tensor or LieTensor inputs
            and parameters and returns a tuple of Tensors/LieTensors or a Tensor/LieTensor.
        dampening (float): Levenberg's dampening factor (non-negative number) to prevent
            singularity.
        maximize (bool, optional): maximize the params based on the objective, instead of
            minimizing (default: False)
    '''
    def __init__(self, model, dampening, maximize=False):
        self.model = model
        assert dampening >= 0, ValueError("Invalid dampening value: {}".format(dampening))
        defaults = dict(dampening=dampening, maximize=maximize)
        super().__init__(model.parameters(), defaults)

    @torch.no_grad()
    def step(self, closure=None, inputs=None, loss=None):
        r'''
        Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model and returns the
                loss. Defaults to ``None``. Either closure or loss cannot be ``None``.
            inputs (tuple of Tensors/LieTensors or Tensor/LieTensor): inputs to the model. Defaults
                to ``None``. Cannot be ``None`` if the model requires inputs.
            loss (tuple of Tensors/LieTensors or Tensor/LieTensor): error of the model to an
                objective. Defaults to ``None``. If sets to ``None``, then model error will be
                computed by calling ``closure``.

        Note:
            Different from PyTorch optimizers like
            `SGD <https://pytorch.org/docs/stable/generated/torch.optim.SGD.html>`_, where the model
            loss has to be a scalar, the model loss of :obj:`LM`  can be a Tensor/LieTensor or a
            tuple of Tensors/LieTensors.

        Example:

        '''
        loss = closure() if loss is None else loss
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
            [p.add_(maximize * (d@loss.view(-1, 1)).view(p.shape))  \
                        for p, d in zip(group['params'], D) if p.requires_grad]
