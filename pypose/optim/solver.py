import torch, warnings
from torch import Tensor, nn
from torch.linalg import pinv, lstsq, cholesky_ex


class PINV(nn.Module):
    r"""The batched linear solver with pseudo inversion.

    The linear equation:

    .. math::
        \mathbf{A}_i \bm{x}_i = \mathbf{b}_i,

    where :math:`\mathbf{A}_i \in \mathbb{C}^{M \times N}` and :math:`\bm{b}_i \in \mathbb{C}^{M \times 1}`
    are the :math:`i`-th item of batched linear equations.

    The solution is given by calculating the pseudo inversion of the matrix :math:`\mathbf{A}`, i.e.,

    .. math::
         \bm{x}_i = \mathrm{pinv}(\mathbf{A}_i) \mathbf{b}_i,
        
    where :math:`\mathrm{pinv}` is the pseudo inversion function.

    Args:
        atol (float, Tensor, optional): the absolute tolerance value. When None it's considered to be
            zero. Default: None.

        rtol (float, Tensor, optional): the relative tolerance value. See above for the value it takes
            when None. Default: None.

        hermitian (bool, optional): indicates whether A is Hermitian if complex or symmetric if real.
            Default: False.
    """
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
    r"""The batched linear solver with pseudo inversion.

    The linear equation:

    .. math::
        \mathbf{A}_i \bm{x}_i = \mathbf{b}_i,

    where :math:`\mathbf{A}_i \in \mathbb{C}^{M \times N}` and :math:`\bm{b}_i \in \mathbb{C}^{M \times 1}`
    are the :math:`i`-th item of batched linear equations.

    The solution is given by calculating the pseudo inversion of the matrix :math:`\mathbf{A}`, i.e.,

    .. math::
         \bm{x}_i = \mathrm{pinv}(\mathbf{A}_i) \mathbf{b}_i,
        
    where :math:`\mathrm{pinv}` is the pseudo inversion function.

    Args:
        rcond (float, optional): used to determine the effective rank of :math:`\mathbf{A}`. It is
            used only when the fast model is enabled. If ``None``, rcond is set to the machine
            precision of the dtype of :math:`\mathbf{A}`. Default: ``None``.
        driver (string, optional): chooses the LAPACK/MAGMA function that will be used. It is
            used only when the fast model is enabled. For CPU users, the valid values are ``gels``,
            ``gelsy``, ``gelsd``, ``gelss``. For CUDA users, the only valid driver is ``gels``,
            which assumes that :math:`\mathbf{A}` is full-rank. If ``None``, ``gelsy`` is used for
            CPU inputs and ``gels`` for CUDA inputs. Default: ``None``.
            To choose the best driver on CPU consider:

            - If :math:`\mathbf{A}` is well-conditioned (its `condition number
              <https://en.wikipedia.org/wiki/Condition_number>`_ is not too large), or you do not
              mind some precision loss.

                - For a general matrix: ``gelsy`` (QR with pivoting) (default)

                - If A is full-rank: ``gels`` (QR)

            - If :math:`\mathbf{A}` is not well-conditioned.

                - ``gelsd`` (tridiagonal reduction and SVD)

                - But if you run into memory issues: ``gelss`` (full SVD).

            See full description of `drivers <https://www.netlib.org/lapack/lug/node27.html>`_.

    See more details of `pseudo inversion
    <https://pytorch.org/docs/stable/generated/torch.linalg.lstsq.html>`_ using
    :obj:`torch.linalg.lstsq`.
    """
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
        x = lstsq(A, b, rcond=self.rcond, driver=self.driver).solution
        assert not torch.any(torch.isnan(x)), \
            'Linear Solver Failed Using LSTSQ. Using PINV() instead'
        return x


class Cholesky(nn.Module):
    r"""The batched linear solver with pseudo inversion.

    The linear equation:

    .. math::
        \mathbf{A}_i \bm{x}_i = \mathbf{b}_i,

    where :math:`\mathbf{A}_i \in \mathbb{C}^{M \times N}` and :math:`\bm{b}_i \in \mathbb{C}^{M \times 1}`
    are the :math:`i`-th item of batched linear equations.

    The solution is given by calculating the pseudo inversion of the matrix :math:`\mathbf{A}`, i.e.,

    .. math::
         \bm{x}_i = \mathrm{pinv}(\mathbf{A}_i) \mathbf{b}_i,
        
    where :math:`\mathrm{pinv}` is the pseudo inversion function.

    Args:
        atol (float, Tensor, optional): the absolute tolerance value. When None it's considered to be
            zero. Default: None.

        rtol (float, Tensor, optional): the relative tolerance value. See above for the value it takes
            when None. Default: None.

        hermitian (bool, optional): indicates whether A is Hermitian if complex or symmetric if real.
            Default: False.
    """
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
