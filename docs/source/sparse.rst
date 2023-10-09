.. automodule:: pypose.sparse

.. currentmodule:: pypose

.. _sparse-docs:

sparse
=============

.. warning::

   This is an experimental feature and subject to change.
   Please feel free to open an issue to report a bug or if you have feedback to share.

Why and when to use sparse block tensor
+++++++++++++++++++++++++++++++++++++++

* By default, ``torch.Tensor`` either stores elements contiguously
  physical memory through a strided format or store sparse elements through the interfaces
  offered in ``torch.sparse``.
* Similar to ``torch.sparse``, ``SbkTensor`` is meant to represent tensors with
  substantial zero or negligible values.
* ``SbkTensor`` is a data structure meant to efficiently represent matrices
  where values are contiguous in small blocks, but blocks are sparsely distributed.
  In applications like SLAM, they play an instrumental role in large-scale bundle
  adjustment.

``SbkTensor`` is capable of representing tensor of any dimension with such sparsity
pattern, as indicated in the dense tensor in the figure.
We next describe the details of the representation (sparsity).

.. image :: https://tinyurl.com/sbk-representation

Format overview
+++++++++++++++

A ``SbkTensor`` object contains a proxy and storage tensor attribute.

- The storage is a
  `Sparse hybrid COO tensor <https://tinyurl.com/sparse-hybrid-coo-tensors>`__,
  which is used to store the values of dense blocks.
- The proxy is a
  `Sparse COO tensor <https://pytorch.org/docs/stable/sparse.html#sparse-coo-tensors>`__,
  which is used to record the spatial organization of dense blocks. It consists of binary
  values to indicate the existence of a dense block at a particular location.

The most common use case is its representation of 2 dimensional dense matrices, as shown
in figure (left), where we want to represent a dense matrix with shape (9, 9). The most
suitable shape of each block is (3, 3), and thus the ``SbkTensor`` will have the
following internal attributes:

- A sparse COO proxy with ``dim=2`` and size of (3, 3). It carries the block pattern
  information, where each non-zero element in the (3, 3) sparse tensor represents the
  existence of a dense block in the ``SbkTensor``. The size of the proxy is irrelevant to
  the size of the block size.

- A sparse hybrid COO storage, with ``sparse_dim=2`` and ``dense_dim=2`` for a total of
  (2 + 2) dimension. The sparse dimension is the same as the proxy, and the dense
  dimension is the same as each of the individual dense block. The ``sparse_dim`` and
  ``dense_dim`` are always equal to the dimention of the original dense tensor. The
  complete ``.shape`` attribute of the storage is (3, 3, 3, 3). The first (3, 3) is the
  size of the proxy, and the second (3, 3) is the size of each dense block.

Note that we are following PyTorch's notation in all descriptions involving shape and dim
of sparse tensors.

Our design is generalized to represent tensors with arbitrary high dimension. Devices such
as GPUs require batching for optimal performance and thus we want to support batch
dimensions, even when the tensor is sparse. For example, Figure (right) shows an example
of batching, where it stacks two 2-D matrices into a single 3-D tensor. The added dimension
is called the batch dimension, with length of 2.
Generally, to represent an n-dimensional dense tensor, ``SbkTensor`` will have an
n-dimensional proxy, and a storage with n-sparse-dimension and n-dense dimension.

Note that the conversion happens on the right-hand side could also result in a proxy with
shape (1, 3, 3) and a storage with shape (1, 3, 3, 2, 3, 3), aside from the one
demonstrated in figure. The proxy is allowed to have length of 1 on the batch dimension
only if the stacked 2-D matrices have the exact same proxy, i.e., spatial element layout.
It is up to the user to decide what shape the proxy shall use exactly, whenever they
instantiate a ``SbkTensor``.

Functionality overview
++++++++++++++++++++++

In the next example we convert a 2D Tensor with default dense (strided)
layout to a 2D Tensor backed by the ``SbkTensor`` layout. Only values and
spatial layout of non-zero elements are required.



Operator overview
+++++++++++++++++
We follow the usage of ``torch.sparse`` as much as possible.
Fundamentally, operations on any sparse tensors behave the same as
operations on dense tensor formats. The particularities of
storage, that is the physical layout of the data, influence the performance of
an operation but should not influence the semantics.


.. _sparse-ops-docs:

Supported operations
++++++++++++++++++++

.. Linear Algebra operations
.. -------------------------

.. The following table summarizes supported Linear Algebra operations on
.. sparse matrices where the operands layouts may vary. Here
.. ``T[layout]`` denotes a tensor with a given layout. Similarly,
.. ``M[layout]`` denotes a matrix (2-D PyTorch tensor), and ``V[layout]``
.. denotes a vector (1-D PyTorch tensor). In addition, ``f`` denotes a
.. scalar (float or 0-D PyTorch tensor), ``*`` is element-wise
.. multiplication, and ``@`` is matrix multiplication.

.. .. csv-table::
..    :header: "PyTorch operation", "Sparse grad?", "Layout signature"
..    :widths: 40, 5, 40
..    :delim: ;

..    :func:`torch.mv`;no; ``M[sparse_coo] @ V[strided] -> V[strided]``
..    :func:`torch.mv`;no; ``M[sparse_csr] @ V[strided] -> V[strided]``
..    :func:`torch.matmul`; no; ``M[sparse_coo] @ M[strided] -> M[strided]``
..    :func:`torch.matmul`; no; ``M[sparse_csr] @ M[strided] -> M[strided]``
..    :func:`torch.matmul`; no; ``M[SparseSemiStructured] @ M[strided] -> M[strided]``
..    :func:`torch.matmul`; no; ``M[strided] @ M[SparseSemiStructured] -> M[strided]``
..    :func:`torch.mm`; no; ``M[sparse_coo] @ M[strided] -> M[strided]``
..    :func:`torch.mm`; no; ``M[SparseSemiStructured] @ M[strided] -> M[strided]``
..    :func:`torch.mm`; no; ``M[strided] @ M[SparseSemiStructured] -> M[strided]``
..    :func:`torch.sparse.mm`; yes; ``M[sparse_coo] @ M[strided] -> M[strided]``
..    :func:`torch.smm`; no; ``M[sparse_coo] @ M[strided] -> M[sparse_coo]``
..    :func:`torch.hspmm`; no; ``M[sparse_coo] @ M[strided] -> M[hybrid sparse_coo]``
..    :func:`torch.bmm`; no; ``T[sparse_coo] @ T[strided] -> T[strided]``
..    :func:`torch.addmm`; no; ``f * M[strided] + f * (M[sparse_coo] @ M[strided]) -> M[strided]``
..    :func:`torch.addmm`; no; ``f * M[strided] + f * (M[SparseSemiStructured] @ M[strided]) -> M[strided]``
..    :func:`torch.addmm`; no; ``f * M[strided] + f * (M[strided] @ M[SparseSemiStructured]) -> M[strided]``
..    :func:`torch.sparse.addmm`; yes; ``f * M[strided] + f * (M[sparse_coo] @ M[strided]) -> M[strided]``
..    :func:`torch.sspaddmm`; no; ``f * M[sparse_coo] + f * (M[sparse_coo] @ M[strided]) -> M[sparse_coo]``
..    :func:`torch.lobpcg`; no; ``GENEIG(M[sparse_coo]) -> M[strided], M[strided]``
..    :func:`torch.pca_lowrank`; yes; ``PCA(M[sparse_coo]) -> M[strided], M[strided], M[strided]``
..    :func:`torch.svd_lowrank`; yes; ``SVD(M[sparse_coo]) -> M[strided], M[strided], M[strided]``

.. where "Sparse grad?" column indicates if the PyTorch operation supports
.. backward with respect to sparse matrix argument. All PyTorch operations,
.. except :func:`torch.smm`, support backward with respect to strided
.. matrix arguments.


Tensor methods and sparse
-------------------------

The following Tensor methods are related to SbkTensor:

.. autosummary::
    :toctree: generated
    :nosignatures:

    sparse.SbkTensor.is_sparse
    sparse.SbkTensor.dense_dim
    sparse.SbkTensor.sparse_dim
    sparse.SbkTensor.to_sparse_coo
    sparse.SbkTensor.to_dense
    sparse.SbkTensor.indices

The following Tensor methods support SbkTensor:

:meth:`~torch.Tensor.add`
:meth:`~torch.Tensor.mul`


Other functions
---------------

The following :mod:`torch` functions support sparse block tensors:

:func:`~torch.mm`

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
