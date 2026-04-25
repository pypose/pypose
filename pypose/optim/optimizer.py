import torch
from .. import bmv
from torch import nn, finfo
from .functional import modjac
from .strategy import TrustRegion
from torch.optim import Optimizer
from .solver import PINV, Cholesky
from torch.linalg import cholesky_ex
from .corrector import FastTriggs
import warnings
from functools import partial
import pypose as pp
from .. import _require_backend_attr

jacobian = None
diagonal_op_ = None


def _is_tracking_tensor(value):
    TrackingTensor = _require_backend_attr(
        "bae.autograd.function",
        "TrackingTensor",
        "pypose.optim.LM(..., sparse=True)",
    )
    return isinstance(value, TrackingTensor)


def _load_sparse_backend_globals():
    global jacobian, diagonal_op_
    if jacobian is None:
        jacobian = _require_backend_attr(
            "bae.autograd.graph",
            "jacobian",
            "pypose.optim.LM(..., sparse=True)",
        )
    if diagonal_op_ is None:
        diagonal_op_ = _require_backend_attr(
            "bae.sparse.py_ops",
            "diagonal_op_",
            "pypose.optim.LM(..., sparse=True)",
        )


def _parameter_update_shape(param):
    if param.ndim == 0:
        return param.shape
    if isinstance(param, pp.LieTensor):
        return torch.Size((*param.shape[:-1], int(param.ltype.manifold[0])))
    return param.shape

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
    Then model regression becomes minimizing the output of the standardized model.
    This class is used during optimization but is not designed to expose to PyPose users.
    '''
    def __init__(self, model, kernel=None, auto=False):
        super().__init__()
        self.model = model
        self.kernel = [Trivial()] if kernel is None else kernel

    def flatten_row_jacobian(self, J, params_values):
        if isinstance(J, (tuple, list)):
            J = torch.cat([j.reshape(-1, p.numel()) for j, p in zip(J, params_values)], 1)
        return J

    def normalize_RWJ(self, R, weight, J):
        weight_diag = None
        if weight is not None:
            weight = weight if isinstance(weight, (tuple, list)) else [weight]
            assert len(R)==len(weight)
            weight_diag = []
            for w, r in zip(weight, R):
                ni = r.numel() * w.shape[-1] / w.numel()
                w = w.view(*w.shape, 1, 1) if r.shape[-1] == 1 else w
                ws = w.view(-1, w.shape[-2], w.shape[-1]).split(1, 0)
                ws = [wsi.squeeze(0) for wsi in ws]
                weight_diag += ws * int(ni)
            weight_diag = torch.block_diag(*weight_diag)
        R = [r.reshape(-1) for r in R]
        J = torch.cat(J) if isinstance(J, (tuple, list)) else J
        return torch.cat(R), weight_diag, J

    def forward(self, input, target=None):
        output = self.model_forward(input)
        return self.residuals(output, target)

    def model_forward(self, input):
        if isinstance(input, dict):
            return self.model(**input)
        if isinstance(input, (tuple, list)):
            return self.model(*input)
        else:
            return self.model(input)

    def residual(self, output, target):
        return output if target is None else output - target

    def residuals(self, outputs, targets):
        if isinstance(outputs, (tuple, list)):
            targets = [None] * len(outputs) if targets is None else targets
            return tuple([self.residual(out, targets[i]) for i, out in enumerate(outputs)])
        return tuple([self.residual(outputs, targets)])

    def loss(self, input, target):
        output = self.model_forward(input)
        residuals = self.residuals(output, target)
        if len(self.kernel) > 1:
            residuals = [k(r.square().sum(-1)).sum() for k, r in zip(self.kernel, residuals)]
        else:
            residuals = [self.kernel[0](r.square().sum(-1)).sum() for r in residuals]
        return sum(residuals)


class _Optimizer(Optimizer):
    r'''
    Base class for all second order optimizers.
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update_parameter(self, params, step):
        r'''
        params will be updated by calling this function
        '''
        steps = step.split([p.numel() for p in params if p.requires_grad])
        [p.add_(d.view(p.shape)) for p, d in zip(params, steps) if p.requires_grad]


class GaussNewton(_Optimizer):
    r'''
    The Gauss-Newton (GN) algorithm solving non-linear least squares problems. This implementation
    is for optimizing the model parameters to approximate the target, which can be a
    Tensor/LieTensor or a tuple of Tensors/LieTensors.

    .. math::
        \bm{\theta}^* = \arg\min_{\bm{\theta}} \sum_i
            \rho\left((\bm{f}(\bm{\theta},\bm{x}_i)-\bm{y}_i)^T \mathbf{W}_i
            (\bm{f}(\bm{\theta},\bm{x}_i)-\bm{y}_i)\right),

    where :math:`\bm{f}()` is the model, :math:`\bm{\theta}` is the parameters to be optimized,
    :math:`\bm{x}` is the model input, :math:`\mathbf{W}_i` is a weighted square matrix (positive
    definite), and :math:`\rho` is a robust kernel function to reduce the effect of outliers.
    :math:`\rho(x) = x` is used by default.

    .. math::
       \begin{aligned}
            &\rule{113mm}{0.4pt}                                                                 \\
            &\textbf{input}: \bm{\theta}_0~\text{(params)}, \bm{f}~\text{(model)},
                \bm{x}~(\text{input}), \bm{y}~(\text{target}), \rho~(\text{kernel})              \\
            &\rule{113mm}{0.4pt}                                                                 \\
            &\textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}                         \\
            &\hspace{5mm} \mathbf{J} \leftarrow {\dfrac {\partial \bm{f} } {\partial \bm{\theta}_{t-1}}}             \\
            &\hspace{5mm} \mathbf{R} = \bm{f(\bm{\theta}_{t-1}, \bm{x})}-\bm{y}                  \\
            &\hspace{5mm} \mathbf{R}, \mathbf{J}=\mathrm{corrector}(\rho, \mathbf{R}, \mathbf{J})\\
            &\hspace{5mm} \bm{\delta} = \mathrm{solver}(\mathbf{W}\mathbf{J}, -\mathbf{W}\mathbf{R})                 \\
            &\hspace{5mm} \bm{\theta}_t \leftarrow \bm{\theta}_{t-1} + \bm{\delta}               \\
            &\rule{113mm}{0.4pt}                                                          \\[-1.ex]
            &\bf{return} \:  \theta_t                                                     \\[-1.ex]
            &\rule{113mm}{0.4pt}                                                          \\[-1.ex]
       \end{aligned}

    Args:
        model (nn.Module): a module containing learnable parameters.
        solver (nn.Module, optional): a linear solver. Available linear solvers include
            :meth:`solver.PINV` and :meth:`solver.LSTSQ`. If ``None``, :meth:`solver.PINV` is used.
            Default: ``None``.
        kernel (nn.Module, or :obj:`list`, optional): the robust kernel function. If a :obj:`list`,
            the element must be nn.Module or ``None`` and the length must be 1 or the number of residuals.
            Default: ``None``.
        corrector: (nn.Module, or :obj:`list`, optional): the Jacobian and model residual corrector to
            fit the kernel function. If a :obj:`list`, the element must be nn.Module or ``None`` and
            the length must be 1 or the number of residuals.
            If a kernel is given but a corrector is not specified, auto correction is
            used. Auto correction can be unstable when the robust model has indefinite Hessian.
            Default: ``None``.
        weight (:obj:`Tensor`, or :obj:`list`, optional): the square positive definite matrix defining
            the weight of model residual. If a :obj:`list`, the element must be :obj:`Tensor` and
            the length must be equal to the number of residuals.
            The corresponding residual and weight should be
            `broadcastable <https://pytorch.org/docs/stable/notes/broadcasting.html#broadcasting-semantics>`_.
            For example, if the shape of a residual is ``B*M*N*R``, the shape of its weight can
            be ``R*R``, ``N*R*R``, ``M*N*R*R`` or ``B*M*N*R*R``.
            Use this only when all inputs shared the same weight matrices. This is
            ignored when weight is given when calling :meth:`.step` or :meth:`.optimize` method.
            Default: ``None``.
        vectorize (bool, optional): the method of computing Jacobian. If ``True``, the
            gradient of each scalar in output with respect to the model parameters will be
            computed in parallel with ``"reverse-mode"``. More details go to
            :meth:`pypose.optim.functional.modjac`. Default: ``True``.

    Available solvers: :meth:`solver.PINV`; :meth:`solver.LSTSQ`.

    Available kernels: :meth:`kernel.Huber`; :meth:`kernel.PseudoHuber`; :meth:`kernel.Cauchy`.

    Available correctors: :meth:`corrector.FastTriggs`; :meth:`corrector.Triggs`.

    Warning:
        The output of model :math:`\bm{f}(\bm{\theta},\bm{x}_i)` and target :math:`\bm{y}_i`
        can be any shape, while their **last dimension** :math:`d` is always taken as the
        dimension of model residual, whose inner product is the input to the kernel function.
        This is useful for residuals like re-projection error, whose last dimension is 2.

        Note that **auto correction** is equivalent to the method of 'square-rooting the kernel'
        mentioned in Section 3.3 of the following paper. It replaces the
        :math:`d`-dimensional residual with a one-dimensional one, which loses
        residual-level structural information.

        * Christopher Zach, `Robust Bundle Adjustment Revisited
          <https://link.springer.com/chapter/10.1007/978-3-319-10602-1_50>`_, European
          Conference on Computer Vision (ECCV), 2014.

        **Therefore, the users need to keep the last dimension of model output and target to
        1, even if the model residual is a scalar. If the users flatten all sample residuals
        into a vector (residual inner product will be a scalar), the model Jacobian will be a
        row vector, instead of a matrix, which loses sample-level structural information,
        although computing Jacobian vector is faster.**

    Note:
        Instead of solving :math:`\mathbf{J}^T\mathbf{J}\delta = -\mathbf{J}^T\mathbf{R}`, we solve
        :math:`\mathbf{J}\delta = -\mathbf{R}` via pseudo inversion, which is more numerically
        advisable. Therefore, only solvers with pseudo inversion (inverting non-square matrices)
        such as :meth:`solver.PINV` and :meth:`solver.LSTSQ` are available.
        More details are in Eq. (5) of the paper "`Robust Bundle Adjustment Revisited`_".
    '''
    def __init__(self, model, solver=None, kernel=None, corrector=None, weight=None, vectorize=True):
        super().__init__(model.parameters(), defaults={})
        self.jackwargs = {'vectorize': vectorize}
        self.solver = PINV() if solver is None else solver
        self.weight = weight
        if kernel is not None:
            kernel = [kernel] if not isinstance(kernel, (tuple, list)) else kernel
            kernel = [k if k is not None else Trivial() for k in kernel]
            self.corrector = [FastTriggs(k) for k in kernel] if corrector is None else corrector
        else:
            self.corrector = [Trivial()] if corrector is None else corrector
        self.corrector = [self.corrector] if not isinstance(self.corrector, (tuple, list)) else self.corrector
        self.corrector = [c if c is not None else Trivial() for c in self.corrector]
        self.model = RobustModel(model, kernel)


    @torch.no_grad()
    def step(self, input, target=None, weight=None):
        r'''
        Performs a single optimization step.

        Args:
            input (Tensor/LieTensor, tuple or a dict of Tensors/LieTensors): the input to the model.
            target (Tensor/LieTensor): the model target to approximate.
                If not given, the model output is minimized. Default: ``None``.
            weight (:obj:`Tensor`, or :obj:`list`, optional): the square positive definite matrix defining
                the weight of model residual. If a :obj:`list`, the element must be :obj:`Tensor` and
                the length must be equal to the number of residuals. Default: ``None``.

        Return:
            Tensor: the minimized model loss.

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
            ...     def forward(self, input):
            ...         # the last dimension of the output is 6,
            ...         # which will be the residual dimension.
            ...         return (self.pose.Exp() @ input).Log()
            ...
            >>> posinv = PoseInv(2, 2)
            >>> input = pp.randn_SE3(2, 2)
            >>> optimizer = pp.optim.GN(posinv)
            ...
            >>> for idx in range(10):
            ...     error = optimizer.step(input)
            ...     print('Pose Inversion error %.7f @ %d it'%(error, idx))
            ...     if error < 1e-5:
            ...         print('Early Stopping with error:', error.item())
            ...         break
            ...
            Pose Inversion error: 1.6865690 @ 0 it
            Pose Inversion error: 0.1065131 @ 1 it
            Pose Inversion error: 0.0002673 @ 2 it
            Pose Inversion error: 0.0000005 @ 3 it
            Early Stopping with error: 5.21540641784668e-07
        '''
        for pg in self.param_groups:
            weight = self.weight if weight is None else weight
            R = list(self.model(input, target))
            J = modjac(self.model, input=(input, target), flatten=False, **self.jackwargs)
            params = dict(self.model.named_parameters())
            params_values = tuple(params.values())
            J = [self.model.flatten_row_jacobian(Jr, params_values) for Jr in J]
            for i in range(len(R)):
                R[i], J[i] = self.corrector[0](R = R[i], J = J[i]) if len(self.corrector) ==1 \
                    else self.corrector[i](R = R[i], J = J[i])
            R, weight, J = self.model.normalize_RWJ(R, weight, J)
            A, b = (J, -R) if weight is None else (weight @ J, -weight @ R)
            D = self.solver(A = A, b = b.view(-1, 1))
            self.last = self.loss if hasattr(self, 'loss') \
                        else self.model.loss(input, target)
            self.update_parameter(params = pg['params'], step = D)
            self.loss = self.model.loss(input, target)
        return self.loss

class LevenbergMarquardt(_Optimizer):
    r'''
    The Levenberg-Marquardt (LM) algorithm solving non-linear least squares problems. It
    is also known as the damped least squares (DLS) method. This implementation is for
    optimizing the model parameters to approximate the target, which can be a
    Tensor/LieTensor or a tuple of Tensors/LieTensors.

    .. math::
        \bm{\theta}^* = \arg\min_{\bm{\theta}} \sum_i
            \rho\left((\bm{f}(\bm{\theta},\bm{x}_i)-\bm{y}_i)^T \mathbf{W}_i
            (\bm{f}(\bm{\theta},\bm{x}_i)-\bm{y}_i)\right),

    where :math:`\bm{f}()` is the model, :math:`\bm{\theta}` is the parameters to be optimized,
    :math:`\bm{x}` is the model input, :math:`\mathbf{W}_i` is a weighted square matrix (positive
    definite), and :math:`\rho` is a robust kernel function to reduce the effect of outliers.
    :math:`\rho(x) = x` is used by default.

    .. math::
       \begin{aligned}
            &\rule{113mm}{0.4pt}                                                                 \\
            &\textbf{input}: \lambda~\text{(damping)}, \bm{\theta}_0~\text{(params)},
                \bm{f}~\text{(model)}, \bm{x}~(\text{input}), \bm{y}~(\text{target})             \\
            &\hspace{12mm} \rho~(\text{kernel}), \epsilon_{s}~(\text{min}),
                           \epsilon_{l}~(\text{max})                                             \\
            &\rule{113mm}{0.4pt}                                                                 \\
            &\textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}                         \\
            &\hspace{5mm} \mathbf{J} \leftarrow {\dfrac {\partial \bm{f}} {\partial \bm{\theta}_{t-1}}}       \\
            &\hspace{5mm} \mathbf{A} \leftarrow (\mathbf{J}^T \mathbf{W} \mathbf{J})
                                     .\mathrm{diagnal\_clamp(\epsilon_{s}, \epsilon_{l})}        \\
            &\hspace{5mm} \mathbf{R} = \bm{f(\bm{\theta}_{t-1}, \bm{x})}-\bm{y}                               \\
            &\hspace{5mm} \mathbf{R}, \mathbf{J}=\mathrm{corrector}(\rho, \mathbf{R}, \mathbf{J})\\
            &\hspace{5mm} \textbf{while}~\text{first iteration}~\textbf{or}~
                                         \text{loss not decreasing}                              \\
            &\hspace{10mm} \mathbf{A} \leftarrow \mathbf{A} + \lambda \mathrm{diag}(\mathbf{A})  \\
            &\hspace{10mm} \bm{\delta} = \mathrm{solver}(\mathbf{A}, -\mathbf{J}^T \mathbf{W} \mathbf{R})     \\
            &\hspace{10mm} \lambda \leftarrow \mathrm{strategy}(\lambda,\text{model information})\\
            &\hspace{10mm} \bm{\theta}_t \leftarrow \bm{\theta}_{t-1} + \bm{\delta}              \\
            &\hspace{10mm} \textbf{if}~\text{loss not decreasing}~\textbf{and}~
                                       \text{maximum reject step not reached}                    \\
            &\hspace{15mm} \bm{\theta}_t \leftarrow \bm{\theta}_{t-1} - \bm{\delta}
                                                               ~(\text{reject step})             \\
            &\rule{113mm}{0.4pt}                                                          \\[-1.ex]
            &\bf{return} \:  \theta_t                                                     \\[-1.ex]
            &\rule{113mm}{0.4pt}                                                          \\[-1.ex]
       \end{aligned}

    Args:
        model (nn.Module): a module containing learnable parameters.
        solver (nn.Module, optional): a linear solver. If ``None``, :meth:`solver.Cholesky` is used.
            Default: ``None``.
        strategy (object, optional): strategy for adjusting the damping factor. If ``None``, the
            :meth:`strategy.TrustRegion` will be used. Defult: ``None``.
        kernel (nn.Module, or :obj:`list`, optional): the robust kernel function. If a :obj:`list`,
            the element must be nn.Module or ``None`` and the length must be 1 or the number of residuals.
            Default: ``None``.
        corrector: (nn.Module, or :obj:`list`, optional): the Jacobian and model residual corrector to
            fit the kernel function. If a :obj:`list`, the element must be nn.Module or ``None`` and
            the length must be 1 or the number of residuals.
            If a kernel is given but a corrector is not specified, auto correction is
            used. Auto correction can be unstable when the robust model has indefinite Hessian.
            Default: ``None``.
        weight (:obj:`Tensor`, or :obj:`list`, optional): the square positive definite matrix defining
            the weight of model residual. If a :obj:`list`, the element must be :obj:`Tensor` and
            the length must be equal to the number of residuals.
            The corresponding residual and weight should be
            `broadcastable <https://pytorch.org/docs/stable/notes/broadcasting.html#broadcasting-semantics>`_.
            For example, if the shape of a residual is ``B*M*N*R``, the shape of its weight can
            be ``R*R``, ``N*R*R``, ``M*N*R*R`` or ``B*M*N*R*R``.
            Use this only when all inputs shared the same weight matrices. This is
            ignored when weight is given when calling :meth:`.step` or :meth:`.optimize` method.
            Default: ``None``.
        reject (integer, optional): the maximum number of rejecting unsuccessfull steps.
            Default: 16.
        min (float, optional): the lower-bound of the Hessian diagonal. Default: 1e-6.
        max (float, optional): the upper-bound of the Hessian diagonal. Default: 1e32.
        vectorize (bool, optional): the method of computing Jacobian. If ``True``, the
            gradient of each scalar in output with respect to the model parameters will be
            computed in parallel with ``"reverse-mode"``. More details go to
            :meth:`pypose.optim.functional.modjac`. Default: ``True``.
        sparse (bool, optional): if ``True``, use the sparse LM path based on sparse Jacobians
            and sparse normal equations. This mode requires the optional sparse backend `bae`
            and is intended to be used with sparse linear solvers such as :meth:`solver.CG`.
            Default: ``False``.

    Available solvers: :meth:`solver.PINV`; :meth:`solver.LSTSQ`; :meth:`solver.Cholesky`;
    :meth:`solver.CG`; :meth:`solver.PCG`.

    Available kernels: :meth:`kernel.Huber`; :meth:`kernel.PseudoHuber`; :meth:`kernel.Cauchy`.

    Available correctors: :meth:`corrector.FastTriggs`, :meth:`corrector.Triggs`.

    Available strategies: :meth:`strategy.Constant`; :meth:`strategy.Adaptive`;
    :meth:`strategy.TrustRegion`;

    Note:
        Setting ``sparse=True`` enables the sparse Jacobian / sparse LM backend.
        It should be used when the underlying optimization problem exhibits a large, structured
        sparse Jacobian, where each residual depends only on a small subset of parameters.
        Please cite the following paper implementing the sparse LM backend:

        * Zitong Zhan, Huan Xu, Zihang Fang, Xinpeng Wei, Yaoyu Hu, Chen Wang,
          `Bundle Adjustment in the Eager Mode <https://arxiv.org/abs/2409.12190>`_,
          IEEE Transactions on Robotics, 2026.

        Check a `full and clean runable example
        <https://github.com/pypose/pypose/tree/main/examples/module/ba>`_ with
        ``sparse=True`` for bundle adjustment.

    Warning:
        The output of model :math:`\bm{f}(\bm{\theta},\bm{x}_i)` and target :math:`\bm{y}_i`
        can be any shape, while their **last dimension** :math:`d` is always taken as the
        dimension of model residual, whose inner product will be input to the kernel
        function. This is useful for residuals like re-projection error, whose last
        dimension is 2.

        Note that **auto correction** is equivalent to the method of 'square-rooting the kernel'
        mentioned in Section 3.3 of the following paper. It replaces the
        :math:`d`-dimensional residual with a one-dimensional one, which loses
        residual-level structural information.

        * Christopher Zach, `Robust Bundle Adjustment Revisited
          <https://link.springer.com/chapter/10.1007/978-3-319-10602-1_50>`_, European
          Conference on Computer Vision (ECCV), 2014.

        **Therefore, the users need to keep the last dimension of model output and target to
        1, even if the model residual is a scalar. If the model output only has one dimension,
        the model Jacobian will be a row vector, instead of a matrix, which loses sample-level
        structural information, although computing Jacobian vector is faster.**
    '''
    def __init__(self, model, solver=None, strategy=None, kernel=None, corrector=None, \
                       weight=None, reject=16, min=1e-6, max=1e32, vectorize=True, sparse=False):
        assert min > 0, ValueError("min value has to be positive: {}".format(min))
        assert max > 0, ValueError("max value has to be positive: {}".format(max))
        self.strategy = TrustRegion() if strategy is None else strategy
        defaults = {**{'min':min, 'max':max}, **self.strategy.defaults}
        super().__init__(model.parameters(), defaults=defaults)

        self.sparse = sparse
        if self.sparse:
            _load_sparse_backend_globals()

        self.jackwargs = {'vectorize': vectorize}
        self.solver = Cholesky() if solver is None else solver
        self.reject, self.reject_count = reject, 0
        self.weight = weight
        if kernel is not None:
            kernel = [kernel] if not isinstance(kernel, (tuple, list)) else kernel
            kernel = [k if k is not None else Trivial() for k in kernel]
            self.corrector = [FastTriggs(k) for k in kernel] if corrector is None else corrector
        else:
            self.corrector = [Trivial()] if corrector is None else corrector
        self.corrector = [self.corrector] if not isinstance(self.corrector, (tuple, list)) else self.corrector
        self.corrector = [c if c is not None else Trivial() for c in self.corrector]
        self.model = RobustModel(model, kernel)

    def update_parameter(self, params, step):
        if getattr(self, 'sparse', False):
            numels = []
            for param in params:
                if param.requires_grad:
                    numels.append(_parameter_update_shape(param).numel())
            steps = step.split(numels)
            for (param, d) in zip(params, steps):
                if param.requires_grad:
                    param.add_(d.view(_parameter_update_shape(param)))
        else:
            super().update_parameter(params, step)

    @torch.no_grad()
    def step(self, input, target=None, weight=None):
        r'''
        Performs a single optimization step.

        Args:
            input (Tensor/LieTensor, tuple or a dict of Tensors/LieTensors): the input to the model.
            target (Tensor/LieTensor): the model target to optimize.
                If not given, the squared model output is minimized. Defaults: ``None``.
            weight (:obj:`Tensor`, or :obj:`list`, optional): the square positive definite matrix defining
                the weight of model residual. If a :obj:`list`, the element must be :obj:`Tensor` and
                the length must be equal to the number of residuals. This argument is currently not
                supported when ``sparse=True``. Default: ``None``.

        Return:
            Tensor: the minimized model loss.

        Note:
            The (non-negative) damping factor :math:`\lambda` can be adjusted at each iteration.
            If the residual reduces rapidly, a smaller value can be used, bringing the algorithm
            closer to the Gauss-Newton algorithm, whereas if an iteration gives insufficient
            residual reduction, :math:`\lambda` can be increased, giving a step closer to the
            gradient descent direction.

            See more details of `Levenberg-Marquardt (LM) algorithm
            <https://en.wikipedia.org/wiki/Levenberg-Marquardt_algorithm>`_ on Wikipedia.

        Note:
            Different from PyTorch optimizers like
            `SGD <https://pytorch.org/docs/stable/generated/torch.optim.SGD.html>`_, where the
            model error has to be a scalar, the output of model :math:`\bm{f}` can be a
            Tensor/LieTensor or a tuple of Tensors/LieTensors.

        Note:
            When ``sparse=True``, only a single residual tensor is currently supported. If the
            model returns multiple residuals, only the first one is used.

        Example:
            Optimizing a simple module to **approximate pose inversion**.

            >>> class PoseInv(nn.Module):
            ...     def __init__(self, *dim):
            ...         super().__init__()
            ...         self.pose = pp.Parameter(pp.randn_se3(*dim))
            ...
            ...     def forward(self, input):
            ...         # the last dimension of the output is 6,
            ...         # which will be the residual dimension.
            ...         return (self.pose.Exp() @ input).Log()
            ...
            >>> posinv = PoseInv(2, 2)
            >>> input = pp.randn_SE3(2, 2)
            >>> strategy = pp.optim.strategy.Adaptive(damping=1e-6)
            >>> optimizer = pp.optim.LM(posinv, strategy=strategy)
            ...
            >>> for idx in range(10):
            ...     loss = optimizer.step(input)
            ...     print('Pose Inversion loss %.7f @ %d it'%(loss, idx))
            ...     if loss < 1e-5:
            ...         print('Early Stopping with loss:', loss.item())
            ...         break
            ...
            Pose Inversion error: 1.6600330 @ 0 it
            Pose Inversion error: 0.1296970 @ 1 it
            Pose Inversion error: 0.0008593 @ 2 it
            Pose Inversion error: 0.0000004 @ 3 it
            Early Stopping with error: 4.443569991963159e-07

            Solving bundle adjustment by minimizing reprojection error with **sparse LM**.
            (requires the optional sparse backend `bae` and CUDA).

            Here, psjac marks the reprojection residual so the
            sparse backend can assemble sparse Jacobians for sparse LM. Use it on factorwise
            residual functions that take batch inputs and return one residual block per
            batch item. When you call the function normally, it behaves the same as before;
            the decorator only helps the sparse backend build sparse Jacobians.
            In the example, both camera poses and 3D points are optimized to minimize
            the reprojection error.

            >>> import torch
            >>> import pypose as pp
            >>> from torch import nn
            >>> from pypose.optim import LM
            >>> from pypose.optim.solver import PCG
            >>> from pypose.optim.strategy import TrustRegion
            >>> from pypose.optim.scheduler import StopOnPlateau
            >>> from pypose.autograd.function import psjac
            >>>
            >>> class ReprojErr(nn.Module):
            ...     def __init__(self, poses, points):
            ...         super().__init__()
            ...         # sjac: enable tracing sparse Jacobian
            ...         self.poses = pp.Parameter(poses, sjac=True)
            ...         self.points = pp.Parameter(points, sjac=True)
            ...
            ...     @psjac  # parallelize assembly of sparse Jacobian
            ...     def project(poses, points):
            ...         points = poses.Act(points)
            ...         return - points[..., :2] / points[..., [2]]
            ...
            ...     def forward(self, pixels, cidx, pidx):
            ...         poses = self.poses[cidx]
            ...         points = self.points[pidx]
            ...         return ReprojErr.project(poses, points) - pixels
            ...
            >>> torch.set_default_device("cuda")
            >>> npts, poses = 8, pp.randn_SE3(1)
            >>> points = torch.randn(npts, 3)
            >>> points[:, 2] += 4  # positive depth
            >>> cidx = torch.zeros(npts, dtype=torch.long)
            >>> pidx = torch.arange(npts)
            >>> pixels = torch.randn(npts, 2)
            >>> inputs = (pixels, cidx, pidx)
            >>>
            >>> model = ReprojErr(poses, points)
            >>> strategy = TrustRegion(up=2.0, down=0.5**4)
            >>> solver = PCG(tol=1e-4, maxiter=250)
            >>> optimizer = LM(model, solver, strategy, sparse=True)
            >>> scheduler = StopOnPlateau(optimizer, steps=5, verbose=True)
            >>>
            >>> while scheduler.continual():
            ...     loss = optimizer.step(inputs)
            ...     scheduler.step(loss)
            StopOnPlateau on step 1 Loss 4.467577e+01 --> Loss 7.737489e+00.
            StopOnPlateau on step 2 Loss 7.737489e+00 --> Loss 2.681350e+00.
            StopOnPlateau on step 3 Loss 2.681350e+00 --> Loss 1.192892e+00.
            StopOnPlateau on step 4 Loss 1.192892e+00 --> Loss 5.546592e-01.
            StopOnPlateau on step 5 Loss 5.546592e-01 --> Loss 5.288146e-01.
            StopOnPlateau: Maximum steps reached, Quitting..
        '''
        for pg in self.param_groups:
            if self.sparse:
                assert weight is None, "Weight is not supported in sparse mode for now."
                R = self.model(input, target)
                if isinstance(R, (tuple, list)):
                    if len(R) > 1:
                        warnings.warn("Sparse mode only supports a single residual. Using the first one.")
                    R = R[0]

                J = jacobian(R, pg['params'])
                if _is_tracking_tensor(R):
                    R = R.tensor()
                J = torch.cat([j.to_sparse_coo() for j in J], dim=-1).to_sparse_csr()
                J_T = J.mT.to_sparse_csr()
                A = J_T @ J
                diagonal_op_(A, op=partial(torch.clamp_, min=pg['min'], max=pg['max']))
            else:
                weight = self.weight if weight is None else weight
                R = list(self.model(input, target))
                J = modjac(self.model, input=(input, target), flatten=False, **self.jackwargs)
                params = dict(self.model.named_parameters())
                params_values = tuple(params.values())
                J = [self.model.flatten_row_jacobian(Jr, params_values) for Jr in J]
                for i in range(len(R)):
                    R[i], J[i] = self.corrector[0](R = R[i], J = J[i]) if len(self.corrector) ==1 \
                        else self.corrector[i](R = R[i], J = J[i])
                R, weight, J = self.model.normalize_RWJ(R, weight, J)
                J_T = J.T @ weight if weight is not None else J.T
                A = J_T @ J
                A.diagonal().clamp_(pg['min'], pg['max'])

            self.last = self.loss = self.loss if hasattr(self, 'loss') \
                                    else self.model.loss(input, target)
            self.reject_count = 0
            while self.last <= self.loss:
                if self.sparse:
                    diagonal_op_(A, op=partial(torch.mul, other=1 + pg['damping']))
                else:
                    A.diagonal().add_(A.diagonal() * pg['damping'])
                try:
                    D = self.solver(A = A, b = -J_T @ R.view(-1, 1))
                except Exception as e:
                    print(e, "\nLinear solver failed. Breaking optimization step...")
                    break
                self.update_parameter(pg['params'], D)
                self.loss = self.model.loss(input, target)
                self.strategy.update(pg, last=self.last, loss=self.loss, J=J, D=D, R=R.view(-1, 1))
                if self.last < self.loss and self.reject_count < self.reject: # reject step
                    self.update_parameter(params = pg['params'], step = -D)
                    self.loss, self.reject_count = self.last, self.reject_count + 1
                else:
                    break
        return self.loss
