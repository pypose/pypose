import torch, warnings
from torch import Tensor, nn
from torch.linalg import pinv, lstsq, cholesky_ex


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
