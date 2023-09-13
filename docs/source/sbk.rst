.. automodule:: pypose.sparse

.. currentmodule:: pypose

.. _sparse-docs:

pypose.sparse
============

.. warning::

   This is an experimental feature and subject to change.
   Please feel free to open an issue to report a bug or if you have feedback to share.

Why and when to use sparse block tensor
+++++++++++++++++++++++++++++++++++++++

By default :class:`torch.Tensor` either stores elements contiguously
physical memory through a strided format or store sparse elements through the interfaces
offered in `torch.sparse`.
Similar to `torch.sparse`, sparse block tensors are meant to represent tensors with
substantial zero or negligible values.
However, it is a data structure meant to efficiently represent matrices where values are
contiguous in small blocks but blocks are sparsely distributed.
In applications like SLAM, they play an instrumental role in large-scale bundle adjustment.

Our SbkTensor is capable of representing tensor of any dimensionality with such sparsity pattern, as indicated in the dense tensor in the figure. We next describe the details of the representation (sparsification).
.. image :: https://user-images.githubusercontent.com/24406547/236364949-2b0c3a20-dcaf-4d34-b311-8be244f6d39f.png
   :alt: sparse block tensor
   :align: center

Format overview
+++++++++++++++
.. Sparse-hybrid-coo-tensors
.. https://pytorch.org/docs/stable/sparse.html#sparse-hybrid-coo-tensors

A SbkTensor object contains a proxy and stroage tensor attribute.
The storage is a `Sparse hybrid COO tensor <https://pytorch.org/docs/stable/sparse.html#sparse-hybrid-coo-tensors>`__, which is used to store the values of dense blocks.
The proxy is a `Sparse COO tensor <https://pytorch.org/docs/stable/sparse.html#sparse-coo-tensors>__`, which is used to record the spatial organization of dense blocks.

The most common use case is its representation of 2 dimensional dense matrix, as shown in figure (left), where we want to represent a dense matrix with shape (9, 9). The most suitable shape of each block is (3, 3), and thus the resulting SbkTensor will have the following internal attributes:
- a sparse COO proxy with dim=2 and size of (3, 3). It carries the block pattern information, where each non-zero element in the (3, 3) sparse tensor represents the existence of a dense block in the SbkTensor. The size of the proxy is irrelevant to the size of the block size.
- a sparse hybrid COO storage, with sparse_dim=2 and dense_dim=2 for a total of (2+2) dimension. The sparse dimension is the same as the proxy, and the dense dimension is the same as each of the individual dense blocks. The sparse_dim and dense_dim should always be equal to the dim of the original dense tensor. The complete .shape attribute of the storage is (3, 3, 3, 3). The first (3, 3) is the size of the proxy, and the second (3, 3) is the size of each dense block.

Note that we are following PyTorch's notation in all descriptions involving shape and dim of sparse tensors.

Our design is generalized to represent tensors with arbitrary high dimension. Devices such as GPUs require batching for optimal performance and thus we want to support batch dimensions, even when the tensor is sparse, see [source](https://pytorch.org/docs/stable/sparse.html#functionality-overview). For example, Figure (right) shows an example of batching, where it stacks two 2-d matrix into a single 3-d tensor. The added dimension is called the batch dimension, with length of 2.
Generally, to represent an n-dimensional dense tensor, the SBT will have an n-dimensional proxy, and an storage with n-sparse-dimension and n-dense dimension.

Note that the conversion happens on the right hand side could also results in a proxy with shape `(1, 3, 3)` and a storage with shape `(1, 3, 3, 2, 3, 3)`, aside from the one demonstrated  in figure. The proxy is allowed to have length of 1 on the batch dimension only if the stacked 2-d matrices have the exact same proxy (i.e. spatial element layout). It is up to the user to decide what shape the proxy shall use exactly, whenever they instantiate a SbkTensor tensor.

Functionality overview
++++++++++++++++++++++

In the next example we convert a 2D Tensor with default dense (strided)
layout to a 2D Tensor backed by the SbkTensor layout. Only values and
spatial layout of non-zero elements are required.



Operator overview
+++++++++++++++++
We follow the usage of `torch.sparse` as much as possible.
Fundamentally, operations on any sparse tensors behave the same as
operations on dense tensor formats. The particularities of
storage, that is the physical layout of the data, influences the performance of
an operation but should not influence the semantics.


.. _sparse-ops-docs:

Supported operations
+++++++++++++++++++++++++++++++++++

Linear Algebra operations
-------------------------

The following table summarizes supported Linear Algebra operations on
sparse matrices where the operands layouts may vary. Here
``T[layout]`` denotes a tensor with a given layout. Similarly,
``M[layout]`` denotes a matrix (2-D PyTorch tensor), and ``V[layout]``
denotes a vector (1-D PyTorch tensor). In addition, ``f`` denotes a
scalar (float or 0-D PyTorch tensor), ``*`` is element-wise
multiplication, and ``@`` is matrix multiplication.

.. csv-table::
   :header: "PyTorch operation", "Sparse grad?", "Layout signature"
   :widths: 20, 5, 60
   :delim: ;

   :func:`torch.mv`;no; ``M[sparse_coo] @ V[strided] -> V[strided]``
   :func:`torch.mv`;no; ``M[sparse_csr] @ V[strided] -> V[strided]``
   :func:`torch.matmul`; no; ``M[sparse_coo] @ M[strided] -> M[strided]``
   :func:`torch.matmul`; no; ``M[sparse_csr] @ M[strided] -> M[strided]``
   :func:`torch.matmul`; no; ``M[SparseSemiStructured] @ M[strided] -> M[strided]``
   :func:`torch.matmul`; no; ``M[strided] @ M[SparseSemiStructured] -> M[strided]``
   :func:`torch.mm`; no; ``M[sparse_coo] @ M[strided] -> M[strided]``
   :func:`torch.mm`; no; ``M[SparseSemiStructured] @ M[strided] -> M[strided]``
   :func:`torch.mm`; no; ``M[strided] @ M[SparseSemiStructured] -> M[strided]``
   :func:`torch.sparse.mm`; yes; ``M[sparse_coo] @ M[strided] -> M[strided]``
   :func:`torch.smm`; no; ``M[sparse_coo] @ M[strided] -> M[sparse_coo]``
   :func:`torch.hspmm`; no; ``M[sparse_coo] @ M[strided] -> M[hybrid sparse_coo]``
   :func:`torch.bmm`; no; ``T[sparse_coo] @ T[strided] -> T[strided]``
   :func:`torch.addmm`; no; ``f * M[strided] + f * (M[sparse_coo] @ M[strided]) -> M[strided]``
   :func:`torch.addmm`; no; ``f * M[strided] + f * (M[SparseSemiStructured] @ M[strided]) -> M[strided]``
   :func:`torch.addmm`; no; ``f * M[strided] + f * (M[strided] @ M[SparseSemiStructured]) -> M[strided]``
   :func:`torch.sparse.addmm`; yes; ``f * M[strided] + f * (M[sparse_coo] @ M[strided]) -> M[strided]``
   :func:`torch.sspaddmm`; no; ``f * M[sparse_coo] + f * (M[sparse_coo] @ M[strided]) -> M[sparse_coo]``
   :func:`torch.lobpcg`; no; ``GENEIG(M[sparse_coo]) -> M[strided], M[strided]``
   :func:`torch.pca_lowrank`; yes; ``PCA(M[sparse_coo]) -> M[strided], M[strided], M[strided]``
   :func:`torch.svd_lowrank`; yes; ``SVD(M[sparse_coo]) -> M[strided], M[strided], M[strided]``

where "Sparse grad?" column indicates if the PyTorch operation supports
backward with respect to sparse matrix argument. All PyTorch operations,
except :func:`torch.smm`, support backward with respect to strided
matrix arguments.

.. note::

   Currently, PyTorch does not support matrix multiplication with the
   layout signature ``M[strided] @ M[sparse_coo]``. However,
   applications can still compute this using the matrix relation ``D @
   S == (S.t() @ D.t()).t()``.

Tensor methods and sparse
-------------------------

The following Tensor methods are related to sparse tensors:

.. autosummary::
    :toctree: generated
    :nosignatures:

    Tensor.is_sparse
    Tensor.is_sparse_csr
    Tensor.dense_dim
    Tensor.sparse_dim
    Tensor.sparse_mask
    Tensor.to_sparse
    Tensor.to_sparse_coo
    Tensor.to_sparse_csr
    Tensor.to_sparse_csc
    Tensor.to_sparse_bsr
    Tensor.to_sparse_bsc
    Tensor.to_dense
    Tensor.values

The following Tensor methods are specific to sparse COO tensors:

.. autosummary::
    :toctree: generated
    :nosignatures:

    Tensor.coalesce
    Tensor.sparse_resize_
    Tensor.sparse_resize_and_clear_
    Tensor.is_coalesced
    Tensor.indices

The following methods are specific to :ref:`sparse CSR tensors <sparse-csr-docs>` and :ref:`sparse BSR tensors <sparse-bsr-docs>`:

.. autosummary::
    :toctree: generated
    :nosignatures:

    Tensor.crow_indices
    Tensor.col_indices

The following methods are specific to :ref:`sparse CSC tensors <sparse-csc-docs>` and :ref:`sparse BSC tensors <sparse-bsc-docs>`:

.. autosummary::
    :toctree: generated
    :nosignatures:

    Tensor.row_indices
    Tensor.ccol_indices

The following Tensor methods support sparse COO tensors:

:meth:`~torch.Tensor.add`
:meth:`~torch.Tensor.add_`
:meth:`~torch.Tensor.addmm`
:meth:`~torch.Tensor.addmm_`
:meth:`~torch.Tensor.any`
:meth:`~torch.Tensor.asin`
:meth:`~torch.Tensor.asin_`
:meth:`~torch.Tensor.arcsin`
:meth:`~torch.Tensor.arcsin_`
:meth:`~torch.Tensor.bmm`
:meth:`~torch.Tensor.clone`
:meth:`~torch.Tensor.deg2rad`
:meth:`~torch.Tensor.deg2rad_`
:meth:`~torch.Tensor.detach`
:meth:`~torch.Tensor.detach_`
:meth:`~torch.Tensor.dim`
:meth:`~torch.Tensor.div`
:meth:`~torch.Tensor.div_`
:meth:`~torch.Tensor.floor_divide`
:meth:`~torch.Tensor.floor_divide_`
:meth:`~torch.Tensor.get_device`
:meth:`~torch.Tensor.index_select`
:meth:`~torch.Tensor.isnan`
:meth:`~torch.Tensor.log1p`
:meth:`~torch.Tensor.log1p_`
:meth:`~torch.Tensor.mm`
:meth:`~torch.Tensor.mul`
:meth:`~torch.Tensor.mul_`
:meth:`~torch.Tensor.mv`
:meth:`~torch.Tensor.narrow_copy`
:meth:`~torch.Tensor.neg`
:meth:`~torch.Tensor.neg_`
:meth:`~torch.Tensor.negative`
:meth:`~torch.Tensor.negative_`
:meth:`~torch.Tensor.numel`
:meth:`~torch.Tensor.rad2deg`
:meth:`~torch.Tensor.rad2deg_`
:meth:`~torch.Tensor.resize_as_`
:meth:`~torch.Tensor.size`
:meth:`~torch.Tensor.pow`
:meth:`~torch.Tensor.sqrt`
:meth:`~torch.Tensor.square`
:meth:`~torch.Tensor.smm`
:meth:`~torch.Tensor.sspaddmm`
:meth:`~torch.Tensor.sub`
:meth:`~torch.Tensor.sub_`
:meth:`~torch.Tensor.t`
:meth:`~torch.Tensor.t_`
:meth:`~torch.Tensor.transpose`
:meth:`~torch.Tensor.transpose_`
:meth:`~torch.Tensor.zero_`

Torch functions specific to sparse Tensors
------------------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    sparse_coo_tensor
    sparse_csr_tensor
    sparse_csc_tensor
    sparse_bsr_tensor
    sparse_bsc_tensor
    sparse_compressed_tensor
    sparse.sum
    sparse.addmm
    sparse.sampled_addmm
    sparse.mm
    sspaddmm
    hspmm
    smm
    sparse.softmax
    sparse.log_softmax
    sparse.spdiags

Other functions
---------------

The following :mod:`torch` functions support sparse tensors:

:func:`~torch.cat`
:func:`~torch.dstack`
:func:`~torch.empty`
:func:`~torch.empty_like`
:func:`~torch.hstack`
:func:`~torch.index_select`
:func:`~torch.is_complex`
:func:`~torch.is_floating_point`
:func:`~torch.is_nonzero`
:func:`~torch.is_same_size`
:func:`~torch.is_signed`
:func:`~torch.is_tensor`
:func:`~torch.lobpcg`
:func:`~torch.mm`
:func:`~torch.native_norm`
:func:`~torch.pca_lowrank`
:func:`~torch.select`
:func:`~torch.stack`
:func:`~torch.svd_lowrank`
:func:`~torch.unsqueeze`
:func:`~torch.vstack`
:func:`~torch.zeros`
:func:`~torch.zeros_like`

To manage checking sparse tensor invariants, see:

.. autosummary::
    :toctree: generated
    :nosignatures:

    sparse.check_sparse_tensor_invariants

To use sparse tensors with :func:`~torch.autograd.gradcheck` function,
see:

.. autosummary::
    :toctree: generated
    :nosignatures:

    sparse.as_sparse_gradcheck

Unary functions
---------------

We aim to support all zero-preserving unary functions.

If you find that we are missing a zero-preserving unary function
that you need, please feel encouraged to open an issue for a feature request.
As always please kindly try the search function first before opening an issue.

The following operators are supported.

:func:`~pypose.sparse.operations.abs`
:func:`~torch.asin`
:func:`~torch.asinh`
:func:`~torch.atan`
:func:`~torch.atanh`
:func:`~torch.ceil`
:func:`~torch.conj_physical`
:func:`~torch.floor`
:func:`~torch.log1p`
:func:`~torch.neg`
:func:`~torch.round`
:func:`~torch.sin`
:func:`~torch.sinh`
:func:`~torch.sign`
:func:`~torch.sgn`
:func:`~torch.signbit`
:func:`~torch.tan`
:func:`~torch.tanh`
:func:`~torch.trunc`
:func:`~torch.expm1`
:func:`~torch.sqrt`
:func:`~torch.angle`
:func:`~torch.isinf`
:func:`~torch.isposinf`
:func:`~torch.isneginf`
:func:`~torch.isnan`
:func:`~torch.erf`
:func:`~torch.erfinv`
