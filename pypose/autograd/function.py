import torch
from functools import wraps

from .. import _format_sparse_backend_error, _load_optional_backend_attr


_PARALLEL_FOR_SPARSE_JACOBIAN_DOC = r"""
The :func:`parallel_for_sparse_jacobian` decorator allows PyPose's optional
sparse backend to trace and assemble sparse Jacobians more efficiently. It
wraps a batched function whose batch samples are independent, so the function
can be treated as a parallel computation along the batch dimension during
sparse Jacobian construction. Here, the leading dimension of each input and
output tensor is the batch dimension.

.. warning::

   This decorator is required by sparse
   :class:`pypose.optim.LevenbergMarquardt` with ``sparse=True``.

.. admonition:: Example

   .. code-block:: python

      @parallel_for_sparse_jacobian
      def edge_error(node1, node2, relpose):
          # node1: pp.SE3 (N, 7), node2: pp.SE3 (N, 7), relpose: pp.SE3 (N, 7)
          # returns: (N, 6)
          return (relpose.Inv() @ node1.Inv() @ node2).Log().tensor()

   Here, each output is the error for one pose-graph edge, computed
   only from the matching input of ``node1``, ``node2``, and ``relpose``.

.. warning::

   Similar to `torch.vmap <https://docs.pytorch.org/docs/stable/generated/torch.vmap.html>`_,
   it should not be used for functions that mix information across rows, such as
   batch reductions, global statistics, or any computation where one output in the batch
   depends on multiple input batch samples.

.. note::

   This decorator doesn't change the function behavior. It only adds tracing
   information for sparse Jacobian construction.

"""

_TRACKING_TENSOR_DOC = r"""
:class:`TrackingTensor` wraps a tensor (or LieTensor) and
tracks all operations performed on the tensor,
allowing PyPose's sparse backend to build correctly structured sparse Jacobians.

Use it for any parameter that needs to be optimized with the sparse backend,
when the optimization model is instantiated.

.. admonition:: Example

   .. code-block:: python

      import torch
      from torch import nn
      from pypose.autograd.function import TrackingTensor

      class Model(nn.Module):
          def __init__(self, table):
              super().__init__()
              self.table = nn.Parameter(TrackingTensor(table))

          def forward(self, target, idx):
              selected = self.table[idx]
              ones = torch.ones_like(selected)
              features = torch.cat([selected, ones], dim=-1)
              return features - target

   Here, ``self.table`` is the parameter being optimized.
   In ``forward``, the model first selects batch entries from ``self.table`` using ``idx``,
   then concatenates the result with ``ones``, and finally subtracts ``target``.
   ``TrackingTensor`` records this chain of operations
   so the sparse backend knows how the output depends on ``self.table``.
   ``ones`` and ``target`` do not need to be wrapped in ``TrackingTensor``
   because they are fixed values, not optimization variables,
   so their Jacobians are unnecessary.

.. warning::

   Wrap the original batched tensor before performing any operation.
   If you use a regular tensor or LieTensor instead,
   the sparse backend will not recover the Jacobian for the tensor.

.. note::

   :class:`TrackingTensor` does not change numerical results. It only adds
   tracing information for sparse Jacobian construction.

"""


def _missing_tracking_tensor():
    class TrackingTensor(torch.Tensor):
        r"""Placeholder for the optional sparse backend tracking tensor."""

        @staticmethod
        def __new__(cls, data, *args, **kwargs):
            raise ImportError(
                _format_sparse_backend_error("pypose.autograd.function.TrackingTensor")
            )

    TrackingTensor.__doc__ = _TRACKING_TENSOR_DOC
    return TrackingTensor


def _load_tracking_tensor():
    value, _ = _load_optional_backend_attr("bae.autograd.function", "TrackingTensor")
    value = _missing_tracking_tensor() if value is None else value
    value.__module__ = "pypose.autograd"
    value.__doc__ = _TRACKING_TENSOR_DOC
    globals()["TrackingTensor"] = value
    globals()["TT"] = value
    return value


def _missing_parallel_for_sparse_jacobian():
    def parallel_for_sparse_jacobian(function):
        _ = function

        @wraps(function)
        def wrapped(*args, **kwargs):
            raise ImportError(
                _format_sparse_backend_error(
                    "pypose.autograd.function.parallel_for_sparse_jacobian"
                )
            )

        return wrapped

    parallel_for_sparse_jacobian.__doc__ = _PARALLEL_FOR_SPARSE_JACOBIAN_DOC
    return parallel_for_sparse_jacobian


def __getattr__(name):
    if name in {"TT", "TrackingTensor"}:
        return _load_tracking_tensor()
    if name == "parallel_for_sparse_jacobian":
        value, _ = _load_optional_backend_attr("bae.autograd.function", "map_transform")
        value = _missing_parallel_for_sparse_jacobian() if value is None else value
        value.__doc__ = _PARALLEL_FOR_SPARSE_JACOBIAN_DOC
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | {"TT", "TrackingTensor", "parallel_for_sparse_jacobian"})


__all__ = ["TT", "TrackingTensor", "parallel_for_sparse_jacobian"]
