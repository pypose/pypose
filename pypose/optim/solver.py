import torch, warnings
from typing import Optional
from torch import Tensor, nn
from functools import partial
from ..function.linalg import bmv
from torch.linalg import pinv, lstsq, cholesky_ex, vecdot


class PINV(nn.Module):
    r'''The batched linear solver with pseudo inversion.

    .. math::
        \mathbf{A}_i \bm{x}_i = \mathbf{b}_i,

    where :math:`\mathbf{A}_i \in \mathbb{C}^{M \times N}` and :math:`\bm{b}_i \in
    \mathbb{C}^{M \times 1}` are the :math:`i`-th item of batched linear equations.

    The solution is given by

    .. math::
        \bm{x}_i = \mathrm{pinv}(\mathbf{A}_i) \mathbf{b}_i,

    where :math:`\mathrm{pinv}()` is the `pseudo inversion
    <https://en.wikipedia.org/wiki/Moore-Penrose_inverse>`_ function.

    Args:
        atol (float, Tensor, optional): the absolute tolerance value. When None it's considered to
            be zero. Default: ``None``.

        rtol (float, Tensor, optional): the relative tolerance value. Default: ``None``.

        hermitian (bool, optional): indicates whether :math:`\mathbf{A}` is Hermitian if complex or
            symmetric if real. Default: ``False``.

    More details go to `torch.linalg.pinv
    <https://pytorch.org/docs/stable/generated/torch.linalg.pinv.html>`_.

    Warning:
        It is always preferred to use :meth:`LSTSQ`, which is faster and more numerically stable.

    Examples:
        >>> import pypose.optim.solver as ppos
        >>> A, b = torch.randn(2, 3, 3), torch.randn(2, 3, 1)
        >>> solver = ppos.PINV()
        >>> x = solver(A, b)
        tensor([[[-0.2675],
                 [-0.1897],
                 [ 0.2708]],
                [[-0.3472],
                 [ 1.1191],
                 [ 0.3301]]])
    '''
    def __init__(self, atol=None, rtol=None, hermitian=False):
        super().__init__()
        self.atol, self.rtol, self.hermitian = atol, rtol, hermitian

    def forward(self, A: Tensor, b: Tensor) -> Tensor:
        '''
        Args:
            A (Tensor): the input batched tensor.
            b (Tensor): the batched tensor on the right hand side.

        Return:
            Tensor: the solved batched tensor.
        '''
        return pinv(A, atol=self.atol, rtol=self.rtol, hermitian=self.hermitian) @ b



class LSTSQ(nn.Module):
    r'''The batched linear solver with fast pseudo inversion.

    .. math::
        \mathbf{A}_i \bm{x}_i = \mathbf{b}_i,

    where :math:`\mathbf{A}_i \in \mathbb{C}^{M \times N}` and :math:`\bm{b}_i \in
    \mathbb{C}^{M \times 1}` are the :math:`i`-th item of batched linear equations.

    The solution is given by

    .. math::
        \bm{x}_i = \mathrm{lstsq}(\mathbf{A}_i, \mathbf{b}_i),

    where :math:`\mathrm{lstsq}()` computes a solution to the least squares problem
    of a system of linear equations. More details go to `torch.linalg.lstsq
    <https://pytorch.org/docs/stable/generated/torch.linalg.lstsq.html>`_.

    Args:
        rcond (float, optional): Cut-off ratio for small singular values. For the purposes of
            rank determination, singular values are treated as zero if they are smaller than
            rcond times the largest singular value. It is used only when the fast model is
            enabled. If ``None``, rcond is set to the machine precision of the dtype of
            :math:`\mathbf{A}`. Default: ``None``.
        driver (string, optional): chooses the LAPACK/MAGMA function that will be used. It is
            used only when the fast model is enabled. For CPU users, the valid values are ``gels``,
            ``gelsy``, ``gelsd``, ``gelss``. For CUDA users, the only valid driver is ``gels``,
            which assumes that input matrices (:math:`\mathbf{A}`) are full-rank. If ``None``,
            ``gelsy`` is used for CPU inputs and ``gels`` for CUDA inputs. Default: ``None``.
            To choose the best driver on CPU consider:

            - If input matrices (:math:`\mathbf{A}`) are well-conditioned (`condition number
              <https://en.wikipedia.org/wiki/Condition_number>`_ is not too large), or you do not
              mind some precision loss.

                - For a general matrix: ``gelsy`` (QR with pivoting) (default)

                - If A is full-rank: ``gels`` (QR)

            - If input matrices (:math:`\mathbf{A}`) are not well-conditioned.

                - ``gelsd`` (tridiagonal reduction and SVD)

                - But if you run into memory issues: ``gelss`` (full SVD).

            See full description of `drivers <https://www.netlib.org/lapack/lug/node27.html>`_.

    Note:
        This solver is faster and more numerically stable than :meth:`PINV`.

        It is also preferred to use :meth:`Cholesky` if input matrices (:math:`\mathbf{A}`)
        are guaranteed to complex Hermitian or real symmetric positive-definite.

    Examples:
        >>> import pypose.optim.solver as ppos
        >>> A, b = torch.randn(2, 3, 3), torch.randn(2, 3, 1)
        >>> solver = ppos.LSTSQ(driver='gels')
        >>> x = solver(A, b)
        tensor([[[ 0.9997],
                 [-1.3288],
                 [-1.6327]],
                [[ 3.1639],
                 [-0.5379],
                 [-1.2872]]])
    '''
    def __init__(self, rcond=None, driver=None):
        super().__init__()
        self.rcond, self.driver = rcond, driver

    def forward(self, A: Tensor, b: Tensor) -> Tensor:
        '''
        Args:
            A (Tensor): the input batched tensor.
            b (Tensor): the batched tensor on the right hand side.

        Return:
            Tensor: the solved batched tensor.
        '''
        self.out = lstsq(A, b, rcond=self.rcond, driver=self.driver)
        assert not torch.any(torch.isnan(self.out.solution)), \
            'Linear Solver Failed Using LSTSQ. Using PINV() instead'
        return self.out.solution


class Cholesky(nn.Module):
    r'''The batched linear solver with Cholesky decomposition.

    .. math::
        \mathbf{A}_i \bm{x}_i = \mathbf{b}_i,

    where :math:`\mathbf{A}_i \in \mathbb{C}^{M \times N}` and :math:`\bm{b}_i \in
    \mathbb{C}^{M \times 1}` are the :math:`i`-th item of batched linear equations.
    Note that :math:`\mathbf{A}_i` has to be a complex Hermitian or a real symmetric
    positive-definite matrix.

    The solution is given by

    .. math::
        \begin{align*}
            \bm{L}_i &= \mathrm{cholesky}(\mathbf{A}_i), \\
            \bm{x}_i &= \mathrm{cholesky\_solve}(\mathbf{b}_i, \bm{L}_i), \\
        \end{align*}

    where :math:`\mathrm{cholesky}()` is the `Cholesky decomposition
    <https://en.wikipedia.org/wiki/Cholesky_decomposition>`_ function.

    More details go to
    `torch.linalg.cholesky <https://pytorch.org/docs/stable/generated/torch.linalg.cholesky.html>`_
    and
    `torch.cholesky_solve <https://pytorch.org/docs/stable/generated/torch.cholesky_solve.html>`_.

    Args:
        upper (bool, optional): whether use an upper triangular matrix in Cholesky decomposition.
            Default: ``False``.

    Examples:
        >>> import pypose.optim.solver as ppos
        >>> A = torch.tensor([[[1.00, 0.10, 0.00], [0.10, 1.00, 0.20], [0.00, 0.20, 1.00]],
                              [[1.00, 0.20, 0.10], [0.20, 1.00, 0.20], [0.10, 0.20, 1.00]]])
        >>> b = torch.tensor([[[1.], [2.], [3.]], [[1.], [2.], [3.]]])
        >>> solver = ppos.Cholesky()
        >>> x = solver(A, b)
        tensor([[[0.8632],
                 [1.3684],
                 [2.7263]],
                [[0.4575],
                 [1.3725],
                 [2.6797]]])
    '''
    def __init__(self, upper=False):
        super().__init__()
        self.upper = upper

    def forward(self, A: Tensor, b: Tensor) -> Tensor:
        '''
        Args:
            A (Tensor): the input batched tensor.
            b (Tensor): the batched tensor on the right hand side.

        Return:
            Tensor: the solved batched tensor.
        '''
        L, info = cholesky_ex(A, upper=self.upper)
        assert not torch.any(torch.isnan(L)), \
            'Cholesky decomposition failed. Check your matrix (may not be positive-definite)'
        return b.cholesky_solve(L, upper=self.upper)


class CG(nn.Module):
    r'''The batched linear solver with conjugate gradient method.

    .. math::
        \mathbf{A}_i \bm{x}_i = \mathbf{b}_i,

    where :math:`\mathbf{A}_i \in \mathbb{C}^{M \times N}` and :math:`\bm{b}_i \in
    \mathbb{C}^{M \times 1}` are the :math:`i`-th item of batched linear equations.

    This function is a 1:1 replica of `scipy.sparse.linalg.cg <https://docs.scipy.org/doc
    /scipy/reference/generated/scipy.sparse.linalg.cg.html>`_.
    The solution is consistent with the scipy version up to numerical precision.
    Variable names are kept the same as the scipy version for easy reference.
    We recommend using only non-batched or batch size 1 input for this solver, as
    the batched version was not appeared in the original scipy version. When handling
    sparse matrices, the batched computation may introduce additional overhead.

    Examples:
        >>> # dense example
        >>> import pypose.optim.solver as ppos
        >>> A = torch.tensor([[0.1802967, 0.3151198, 0.4548111, 0.3860016, 0.2870615],
                              [0.3151198, 1.4575327, 1.5533425, 1.0540756, 1.0795838],
                              [0.4548111, 1.5533425, 2.3674474, 1.1222278, 1.2365348],
                              [0.3860016, 1.0540756, 1.1222278, 1.3748058, 1.2223261],
                              [0.2870615, 1.0795838, 1.2365348, 1.2223261, 1.2577004]])
        >>> b = torch.tensor([[ 2.64306851],
                              [-0.03593633],
                              [ 0.73612658],
                              [ 0.51501254],
                              [-0.26689271]])
        >>> solver = ppos.CG()
        >>> x = solver(A, b)
        tensor([[246.4098],
                [ 22.6997],
                [-56.9239],
                [-161.7914],
                [137.2683]])

        >>> # sparse csr example
        >>> import pypose.optim.solver as ppos
        >>> crow_indices = torch.tensor([0, 2, 4])
        >>> col_indices = torch.tensor([0, 1, 0, 1])
        >>> values = torch.tensor([1, 2, 3, 4], dtype=torch.float)
        >>> A = torch.sparse_csr_tensor(crow_indices, col_indices, values)
        >>> A.to_dense()  # visualize
        tensor([[1., 2.],
                [3., 4.]])
        >>> b = torch.tensor([[1.], [2.]])
        >>> solver = ppos.CG()
        >>> x = solver(A, b)
        tensor([-4.4052e-05,  5.0003e-01])

    '''
    def __init__(self, maxiter=None, tol=1e-5):
        super().__init__()
        self.maxiter, self.tol = maxiter, tol

    def forward(self, A: Tensor, b: Tensor, x: Optional[Tensor]=None,
                M: Optional[Tensor]=None) -> Tensor:
        '''
        Args:
            A (Tensor): the input tensor. It is assumed to be a symmetric
                positive-definite matrix. Layout is allowed to be COO, CSR, BSR, or dense.
            b (Tensor): the tensor on the right hand side. Layout could be sparse or dense
                but is only allowed to be a type that is compatible with the layout of A.
                In other words, `A @ b` operation must be supported by the layout of A.
            x (Tensor, optional): the initial guess for the solution. Default: ``None``.
            M (Tensor, optional): the preconditioner for A. Layout is allowed to be COO,
                CSR, BSR, or dense. Default: ``None``.

        Return:
            Tensor: the solved tensor. Layout is the same as the layout of b.
        '''
        if A.ndim == b.ndim:
            b = b.squeeze(-1)
        else:
            assert A.ndim == b.ndim + 1, \
                'The number of dimensions of A and b must differ by 1'
        if x is None:
            x = torch.zeros_like(b)
        bnrm2 = torch.linalg.norm(b, dim=-1)
        if (bnrm2 == 0).all():
            return b
        atol = self.tol * bnrm2
        n = b.shape[-1]

        if self.maxiter is None:
            maxiter = n * 10
        else:
            maxiter = self.maxiter
        r = b - bmv(A, x) if x.any() else b.clone()
        rho_prev, p = None, None

        for iteration in range(maxiter):
            if (torch.linalg.norm(r, dim=-1) < atol).all():
                return x

            z = bmv(M, r) if M is not None else r
            rho_cur = vecdot(r, z)
            if iteration > 0:
                beta = rho_cur / rho_prev
                p = p * beta.unsqueeze(-1) + z
            else:  # First spin
                p = torch.empty_like(r)
                p[:] = z[:]

            q = bmv(A, p)
            alpha = rho_cur / vecdot(p, q)
            x += alpha.unsqueeze(-1)*p
            r -= alpha.unsqueeze(-1)*q
            rho_prev = rho_cur

        return x
