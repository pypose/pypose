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
    if name in {"parallel_for_sparse_jacobian", "psjac"}:
        value, _ = _load_optional_backend_attr("bae.autograd.function", "map_transform")
        value = _missing_parallel_for_sparse_jacobian() if value is None else value
        value.__doc__ = _PARALLEL_FOR_SPARSE_JACOBIAN_DOC
        globals()["parallel_for_sparse_jacobian"] = value
        globals()["psjac"] = value
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | {"parallel_for_sparse_jacobian", "psjac"})


__all__ = ["parallel_for_sparse_jacobian", "psjac"]
