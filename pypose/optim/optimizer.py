import torch, warnings
from torch import nn, finfo
from .functional import modjac
from torch.optim import Optimizer
from .solver import PINV, Cholesky


class Trivial(torch.nn.Module):
    r"""
    A trivial module. Get anything, return anything.
    Not supposed to be called by PyPose users.
    """
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, *args, **kwargs):
        out = *args, *kwargs.values()
        return out[0] if len(out) == 1 else out


class RobustModel(nn.Module):
    '''
    Standardize a model for least square problems with an option of square-rooting kernel.
    Then model regression becomes minimizing the outputs of the standardized model.
    This class is used during optimization but is not designed to expose to PyPose users.
    '''
    def __init__(self, model, kernel=None, auto=False):
        super().__init__()
        self.model = model
        self.kernel = Trivial() if kernel is None else kernel
        if auto:
            self.register_forward_hook(self.kernel_forward)

    def forward(self, inputs, targets):
        outputs = self.model_forward(inputs)
        return self.residual(outputs, targets)

    def model_forward(self, inputs):
        if isinstance(inputs, tuple):
            return self.model(*inputs)
        else:
            return self.model(inputs)

    def residual(self, outputs, targets):
        return  outputs if targets is None else targets - outputs

    def kernel_forward(self, module, inputs, outputs):
        # eps is to prevent grad of sqrt() from being inf
        eps = finfo(outputs.dtype).eps
        return self.kernel(outputs.square().sum(-1)).clamp(min=eps).sqrt()

    def loss(self, inputs, targets):
        outputs = self.model_forward(inputs)
        E = self.residual(outputs, targets)
        return self.kernel(E.square().sum(-1)).sum()


class GaussNewton(Optimizer):
    r'''
    The Gauss-Newton (GN) algorithm solving non-linear least squares problems. This implementation
    is for optimizing the model parameters to approximate the targets, which can be a
    Tensor/LieTensor or a tuple of Tensors/LieTensors.

    .. math::
        \bm{\theta}^*=\arg\min_{\bm{\theta}}\sum_i\rho(\|\bm{y}_i-\bm{f}(\bm{\theta},\bm{x}_i)\|^2),

    where :math:`\bm{f}(\bm{\theta}, \bm{x})` is the model, :math:`\bm{\theta}` is the parameters
    to be optimized, :math:`\bm{x}` is the model inputs, and :math:`\rho` is a robust kernel
    function. :math:`\rho(\bm{x})=\bm{x}` is used in default.

    .. math::
       \begin{aligned}
            &\rule{113mm}{0.4pt}                                                                 \\
            &\textbf{input}: \bm{\theta}_0~\text{(params)}, \bm{f}~\text{(model)},
                \bm{x}~(\text{inputs}), \bm{y}~(\text{targets}), \rho~(\text{kernel})            \\
            &\rule{113mm}{0.4pt}                                                                 \\
            &\textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}                         \\
            &\hspace{5mm} \mathbf{J} \leftarrow {\dfrac {\partial \bm{f}}
                {\partial \bm{\theta}_{t-1}}}                                                    \\
            &\hspace{5mm} \mathbf{E} = \bm{y} - \bm{f(\bm{\theta}_{t-1}, \bm{x})}                \\
            &\hspace{5mm} \mathbf{E}, \mathbf{J}=\mathrm{corrector}(\rho, \mathbf{E}, \mathbf{J})\\
            &\hspace{5mm} \bm{\delta} = \mathrm{solver}(\mathbf{J}, -\mathbf{E})                 \\
            &\hspace{5mm} \bm{\theta}_t \leftarrow \bm{\theta}_{t-1} + \bm{\delta}               \\
            &\rule{113mm}{0.4pt}                                                          \\[-1.ex]
            &\bf{return} \:  \theta_t                                                     \\[-1.ex]
            &\rule{113mm}{0.4pt}                                                          \\[-1.ex]
       \end{aligned}

    Args:
        model (nn.Module): a module containing learnable parameters.
        solver (nn.Module, optional): a linear solver. Available linear solvers include
            :meth:`solver.PINV` and :meth:`solver.LSTSQ`. If ``None``, :meth:`solver.PINV` is used.
            Default: None.
        kernel (nn.Module, optional): a robust kernel function. Default: ``None``.
        corrector: (nn.Module, optional): a Jacobian and model residual corrector to fit the kernel
            function. If ``None``, auto correction is used. Auto correction can be unstable when
            the robust model has indefinite Hessian.  Default: ``None``.

    Available solvers: :meth:`solver.PINV`; :meth:`solver.LSTSQ`.

    Available kernels: :meth:`pypose.module.Huber`; :meth:`module.PseudoHuber`; :meth:`module.Cauchy`.

    Available correctors: :meth:`corrector.TrivialScale`, :meth:`corrector.FastTriggs`,
    :meth:`corrector.Triggs`.

    Note:
        Instead of solving :math:`\mathbf{J}^T\mathbf{J}\delta = -\mathbf{J}^T\mathbf{E}`, we solve
        :math:`\mathbf{J}\delta = -\mathbf{E}` via QR-decomposition, which is more numerically
        advisible. Therefore, only solvers with pseudo inversion such as :meth:`solver.PINV` and
        :meth:`solver.LSTSQ` are available. More details are in Eq. (5) of the paper:

        * Christopher Zach, `Robust Bundle Adjustment Revisited
          <https://link.springer.com/chapter/10.1007/978-3-319-10602-1_50>`_, European Conference on
          Computer Vision (ECCV), 2014.
    '''
    def __init__(self, model, solver=None, kernel=None, corrector=None):
        super().__init__(model.parameters(), defaults={})
        self.solver = PINV() if solver is None else solver
        if kernel is not None and corrector is None:
            # auto diff of robust model will be computed
            self.model = RobustModel(model, kernel, auto=True)
            self.corrector = Trivial()
        else:
            # manually Jacobian correction will be computed
            self.model = RobustModel(model, kernel, auto=False)
            self.corrector = Trivial() if corrector is None else corrector

    @torch.no_grad()
    def step(self, inputs, targets=None):
        r'''
        Performs a single optimization step.

        Args:
            inputs (Tensor/LieTensor or tuple of Tensors/LieTensors): the inputs to the model.
            targets (Tensor/LieTensor or tuple of Tensors/LieTensors): the model targets to approximate.
                If not given, the model outputs are minimized. Defaults: ``None``.

        Return:
            Tensor: the minimized model loss, i.e.,
            :math:`\sum_i\rho(\|\bm{y}_i - \bm{f}(\bm{\theta}, \bm{x}_i)\|^2)`.

        Note:
            Different from PyTorch optimizers like
            `SGD <https://pytorch.org/docs/stable/generated/torch.optim.SGD.html>`_, where the model
            error has to be a scalar, the output of model :math:`\bm{f}` can be a Tensor/LieTensor or a
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
        E = self.model(inputs, targets)
        for pg in self.param_groups:
            numels = [p.numel() for p in pg['params'] if p.requires_grad]
            J = modjac(self.model, inputs=(inputs, targets), flatten=True)
            E, J = self.corrector(E = E, J = J)
            D = self.solver(A = J, b = -E.view(-1, 1)).split(numels)
            [p.add_(d.view(p.shape)) for p, d in zip(pg['params'], D) if p.requires_grad]
        return self.model.loss(inputs, targets)


class LevenbergMarquardt(Optimizer):
    r'''
    The Levenberg-Marquardt (LM) algorithm, which is also known as the damped least-squares (DLS)
    method for solving non-linear least squares problems. This implementation is for optimizing the
    model parameters to approximate the targets, which can be a Tensor/LieTensor or a tuple of
    Tensors/LieTensors.

    .. math::
        \bm{\theta}^*=\arg\min_{\bm{\theta}}\sum_i\rho(\|\bm{y}_i-\bm{f}(\bm{\theta},\bm{x}_i)\|^2),

    where :math:`\bm{f}(\bm{\theta}, \bm{x})` is the model, :math:`\bm{\theta}` is the parameters
    to be optimized, :math:`\bm{x}` is the model inputs, and :math:`\rho` is a robust kernel
    function. :math:`\rho(\bm{x})=\bm{x}` is used in default.

    .. math::
       \begin{aligned}
            &\rule{113mm}{0.4pt}                                                                 \\
            &\textbf{input}: \lambda \geq 0~\text{(damping)}, \bm{\theta}_0~\text{(params)},
                \bm{f}~\text{(model)}, \bm{x}~(\text{inputs}), \bm{y}~(\text{targets})           \\
                &\hspace{12mm} \rho~(\text{kernel})                                              \\
            &\rule{113mm}{0.4pt}                                                                 \\
            &\textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}                         \\
            &\hspace{5mm} \mathbf{J} \leftarrow {\dfrac {\partial \bm{f}}
                {\partial \bm{\theta}_{t-1}}}                                                    \\
            &\hspace{5mm} \mathbf{A} \leftarrow (\mathbf{J}^T \mathbf{J} 
                + \lambda \mathrm{diag}(\mathbf{J}^T \mathbf{J})).\mathrm{clamp(min, max)}       \\
            &\hspace{5mm} \mathbf{E} = \bm{y} - \bm{f(\bm{\theta}_{t-1}, \bm{x})}                \\
            &\hspace{5mm} \mathbf{E}, \mathbf{J}=\mathrm{corrector}(\rho, \mathbf{E}, \mathbf{J})\\
            &\hspace{5mm} \bm{\delta} = \mathrm{solver}(\mathbf{A}, -\mathbf{J}^T\mathbf{E})     \\
            &\hspace{5mm} \bm{\theta}_t \leftarrow \bm{\theta}_{t-1} + \bm{\delta}               \\
            &\rule{113mm}{0.4pt}                                                          \\[-1.ex]
            &\bf{return} \:  \theta_t                                                     \\[-1.ex]
            &\rule{113mm}{0.4pt}                                                          \\[-1.ex]
       \end{aligned}

    Args:
        model (nn.Module): a module containing learnable parameters.
        damping (float): Levenberg's damping factor (positive number).
        solver (nn.Module, optional): a linear solver. Available linear solvers include
            :meth:`solver.PINV` and :meth:`solver.LSTSQ`. If ``None``, :meth:`solver.PINV` is used.
            Default: None.
        kernel (nn.Module, optional): a robust kernel function. Default: ``None``.
        corrector: (nn.Module, optional): a Jacobian and model residual corrector to fit the kernel
            function. If ``None``, auto correction is used. Auto correction can be unstable when
            the robust model has indefinite Hessian.  Default: ``None``.
        min (float, optional): the lower-bound of the matrix diagonal to inverse. Default: 1e-6.
        max (float, optional): the upper-bound of the matrix diagonal to inverse. Default: 1e32.

    Available solvers: :meth:`solver.PINV`; :meth:`solver.LSTSQ`, :meth:`solver.Cholesky`.

    Available kernels: :meth:`pypose.module.Huber`; :meth:`module.PseudoHuber`; :meth:`module.Cauchy`.

    Available correctors: :meth:`corrector.TrivialScale`, :meth:`corrector.FastTriggs`,
    :meth:`corrector.Triggs`.
    '''
    def __init__(self, model, damping, solver=None, kernel=None, corrector=None, min=1e-6, max=1e32):
        assert damping > 0, ValueError("damping factor has to be positive: {}".format(damping))
        assert min > 0, ValueError("min value has to be positive: {}".format(min))
        assert max > 0, ValueError("max value has to be positive: {}".format(max))
        defaults = {'damping':damping, 'min':min, 'max':max}
        super().__init__(model.parameters(), defaults=defaults)
        self.solver = Cholesky() if solver is None else solver
        if kernel is not None and corrector is None:
            # auto diff of robust model will be computed
            self.model = RobustModel(model, kernel, auto=True)
            self.corrector = Trivial()
        else:
            # manually Jacobian correction will be computed
            self.model = RobustModel(model, kernel, auto=False)
            self.corrector = Trivial() if corrector is None else corrector

    @torch.no_grad()
    def step(self, inputs, targets=None):
        r'''
        Performs a single optimization step.

        Args:
            inputs (Tensor/LieTensor or tuple of Tensors/LieTensors): the inputs to the model.
            targets (Tensor/LieTensor or tuple of Tensors/LieTensors): the model targets to optimize.
                If not given, the squared model outputs are minimized. Defaults: ``None``.

        Return:
            Tensor: the minimized model loss, i.e.,
            :math:`\sum_i\rho(\|\bm{y}_i - \bm{f}(\bm{\theta}, \bm{x}_i)\|^2)`.

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
            error has to be a scalar, the output of model :math:`\bm{f}` can be a Tensor/LieTensor or a
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
        E = self.model(inputs, targets)
        for pg in self.param_groups:
            numels = [p.numel() for p in pg['params'] if p.requires_grad]
            J = modjac(self.model, inputs=(inputs, targets), flatten=True)
            E, J = self.corrector(E = E, J = J)
            A = J.T @ J
            A.diagonal().add_(pg['damping'] * A.diagonal()).clamp_(pg['min'], pg['max'])
            D = self.solver(A = A, b = -J.T @ E.view(-1, 1)).split(numels)
            [p.add_(d.view(p.shape)) for p, d in zip(pg['params'], D) if p.requires_grad]
        return self.model.loss(inputs, targets)
