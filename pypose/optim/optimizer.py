import torch, warnings
from torch import nn, finfo
from .functional import modjac
from torch.optim import Optimizer
from .solver import PINV, LSTSQ, Cholesky
from .corrector import Trivial, Scale, GradScale


class RobustModel(nn.Module):
    def __init__(self, model, kernel):
        super().__init__()
        self.model = model
        self.kernel = kernel

    def forward(self, *args, targets=None, **kwargs):
        outputs = self.model(*args, **kwargs)
        eps = finfo(outputs.dtype).eps
        # eps is to prevent grad of sqrt() from being -inf
        return (self.kernel(outputs**2) + eps).sqrt()


class _Optimizer(Optimizer):
    r'''
    Base class of 2nd order optimizer.
    '''
    def __init__(self, model, solver, kernel=None, corrector=None, **kwargs):
        super().__init__(model.parameters(), defaults=kwargs)
        self.model, self.solver = model, solver
        if kernel is not None and corrector is None:
            # auto diff of robust model will be computed
            self.rmodel = RobustModel(model, kernel)
            self.kernel, self.corrector = lambda x:x, Trivial()
        else:
            # manually Jacobian correction will be computed
            self.rmodel, self.kernel = model, lambda x:x if kernel is None else kernel
            self.corrector = Trivial() if corrector is None else corrector

    @torch.no_grad()
    def step(self, inputs, targets=None):
        raise NotImplementedError('step method is not implemented.')

    @classmethod
    def _residual(cls, model, inputs, targets=None):
        outputs = model(inputs)
        if targets is not None:
            if isinstance(outputs, tuple):
                E = torch.cat([(t - o).view(-1, 1) for t, o in zip(targets, outputs)])
            else:
                E = (targets - outputs).view(-1, 1)
        else:
            if isinstance(outputs, tuple):
                E = torch.cat([-o.view(-1, 1) for o in outputs])
            else:
                E = -outputs.view(-1, 1)
        return E

    def _loss(self, model, inputs, targets):
        E = self._residual(model, inputs, targets)
        return self.kernel(E**2).sum()


class GaussNewton(_Optimizer):
    r'''
    The Gauss-Newton (GN) algorithm solving non-linear least squares problems. This implementation
    is for optimizing the model parameters to approximate the targets, which can be a
    Tensor/LieTensor or a tuple of Tensors/LieTensors.

    .. math::
        \bm{\theta}^*=\arg\min_{\bm{\theta}}\sum_i\rho(\|\bm{y}_i-\bm{f}(\bm{\theta},\bm{x}_i)\|^2),

    where :math:`\bm{f}(\bm{\theta}, \bm{x})` is the model, :math:`\bm{\theta}` is the parameters
    to be optimized, :math:`\bm{x}` is the model inputs, and :math:`\rho` is a robust kernel
    function. :math:`\rho(\bm{x})=x` is used in default.

    .. math::
       \begin{aligned}
            &\rule{113mm}{0.4pt}                                                                 \\
            &\textbf{input}: \lambda \geq 0~\text{(damping)}, \bm{\theta}_0~\text{(params)},
                \bm{f}~\text{(model)}, \bm{x}~(\text{inputs}), \bm{y}~(\text{targets})           \\
            &\rule{113mm}{0.4pt}                                                                 \\
            &\textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}                         \\
            &\hspace{5mm} \mathbf{J} \leftarrow {\dfrac {\partial \bm{f}}
                {\partial \bm{\theta}_{t-1}}}                                                    \\
            &\hspace{5mm} \mathbf{A} \leftarrow \mathbf{J}^T \mathbf{J}                          \\
            &\hspace{5mm} \mathbf{E} = \bm{y} - \bm{f(\bm{\theta}_{t-1}, \bm{x})}                \\
            &\hspace{5mm} \bm{\delta}=\mathrm{pseudo\_inverse}(\mathbf{A})
                      (\frac{\partial \rho}{\partial \mathbf{E}^2} \cdot \mathbf{J}^T\mathbf{E}) \\
            &\hspace{5mm} \bm{\theta}_t \leftarrow \bm{\theta}_{t-1} + \bm{\delta}               \\
            &\rule{113mm}{0.4pt}                                                          \\[-1.ex]
            &\bf{return} \:  \theta_t                                                     \\[-1.ex]
            &\rule{113mm}{0.4pt}                                                          \\[-1.ex]
       \end{aligned}

    Args:
        model (nn.Module): a module containing learnable parameters.
        solver (nn.Module): a linear solver. If None, :meth:`PINV` will be used. Default: None.
    '''
    def __init__(self, model, solver=None, kernel=None, corrector=None):
        solver = PINV() if solver is None else solver
        super().__init__(model, solver, kernel, corrector)

    @torch.no_grad()
    def step(self, inputs, targets=None):
        r'''
        Performs a single optimization step.

        Args:
            inputs (Tensor/LieTensor or tuple of Tensors/LieTensors): the inputs to the model.
            targets (Tensor/LieTensor or tuple of Tensors/LieTensors): the model targets to approximate.
                If not given, the model outputs are minimized. Defaults: ``None``.

        Return:
            Tensor: the minimized model error, i.e., :math:`\|\bm{y} - \bm{f}(\bm{\theta}, \bm{x})\|^2`.

        Note:
            Different from PyTorch optimizers like
            `SGD <https://pytorch.org/docs/stable/generated/torch.optim.SGD.html>`_, where the model
            error has to be a scalar, the model output of :obj:`LM` can be a Tensor/LieTensor or a
            tuple of Tensors/LieTensors.

            See more details of
            `Gauss-Newton (GN) algorithm <https://en.wikipedia.org/wiki/Gauss-Newton_algorithm>`_ on
            Wikipedia.

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
            >>> optimizer = pp.optim.GN(posinv)
            ...
            >>> for idx in range(10):
            ...     error = optimizer.step(inputs)
            ...     print('Pose Inversion error %.7f @ %d it'%(error, idx))
            ...     if error < 1e-5:
            ...         print('Early Stoping with error:', error.item())
            ...         break
            ...
            Pose Inversion error: 1.6865690 @ 0 it
            Pose Inversion error: 0.1065131 @ 1 it
            Pose Inversion error: 0.0002673 @ 2 it
            Pose Inversion error: 0.0000005 @ 3 it
            Early Stoping with error: 5.21540641784668e-07
        '''
        E = self._residual(self.rmodel, inputs, targets)
        for pg in self.param_groups:
            numels = [p.numel() for p in pg['params'] if p.requires_grad]
            J = modjac(self.rmodel, inputs, flatten=True)
            E, J = self.corrector(E = E, J = J)
            D = self.solver(A = J, b = E).split(numels)
            [p.add_(d.view(p.shape)) for p, d in zip(pg['params'], D) if p.requires_grad]
        return self._loss(self.model, inputs, targets)


class LevenbergMarquardt(_Optimizer):
    r'''
    The Levenberg-Marquardt (LM) algorithm, which is also known as the damped least-squares (DLS)
    method for solving non-linear least squares problems. This implementation is for optimizing the
    model parameters to approximate the targets, which can be a Tensor/LieTensor or a tuple of
    Tensors/LieTensors.

    .. math::
        \bm{\theta}^*=\arg\min_{\bm{\theta}}\sum_i\rho(\|\bm{y}_i-\bm{f}(\bm{\theta},\bm{x}_i)\|^2),

    where :math:`\bm{f}(\bm{\theta}, \bm{x})` is the model, :math:`\bm{\theta}` is the parameters
    to be optimized, :math:`\bm{x}` is the model inputs, and :math:`\rho` is a robust kernel
    function. :math:`\rho(\bm{x})=x` is used in default.

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
            &\hspace{5mm} \bm{\delta}=\mathrm{cholesky\_solve}
              (\frac{\partial \rho}{\partial \mathbf{E}^2} \cdot \mathbf{J}^T \mathbf{E}, \bm{L})\\
            &\hspace{5mm} \bm{\theta}_t \leftarrow \bm{\theta}_{t-1} + \bm{\delta}               \\
            &\rule{113mm}{0.4pt}                                                          \\[-1.ex]
            &\bf{return} \:  \theta_t                                                     \\[-1.ex]
            &\rule{113mm}{0.4pt}                                                          \\[-1.ex]
       \end{aligned}

    Args:
        model (nn.Module): a module containing learnable parameters.
        damping (float): Levenberg's damping factor (positive number).
        kernel (nn.Module, optional): a robust kernel function. Default: ``None``.
        min (float, optional): the lower-bound of the matrix diagonal to inverse.
        max (float, optional): the upper-bound of the matrix diagonal to inverse.
    '''
    def __init__(self, model, damping, solver=None, kernel=None, corrector=None, min=1e-6, max=1e32):
        solver = solver if solver is not None else Cholesky()
        assert damping > 0, ValueError("damping factor has to be positive: {}".format(damping))
        assert min > 0, ValueError("min value has to be positive: {}".format(min))
        assert max > 0, ValueError("max value has to be positive: {}".format(max))
        super().__init__(model, solver, kernel, corrector, damping=damping, min=min, max=min)

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

            See more details of `Levenberg-Marquardt (LM) algorithm
            <https://en.wikipedia.org/wiki/Levenberg-Marquardt_algorithm>`_ on Wikipedia.

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
        E = self._residual(self.rmodel, inputs, targets)
        for pg in self.param_groups:
            numels = [p.numel() for p in pg['params'] if p.requires_grad]
            J = modjac(self.rmodel, inputs, flatten=True)
            E, J = self.corrector(E = E, J = J)
            A = J.T @ J
            A.diagonal().add_(pg['damping'] * A.diagonal().clamp(pg['min'], pg['max']))
            D = self.solver(A = A, b = J.T @ E).split(numels)
            [p.add_(d.view(p.shape)) for p, d in zip(pg['params'], D) if p.requires_grad]
        return self._loss(self.model, inputs, targets)
