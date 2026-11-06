import torch
from .. import LieTensor, retain_ltype
from torch.autograd.functional import jacobian as torch_jacobian
from typing import Callable, Union, Tuple, Optional


def _slice_lie_jacobian(jacobian, value):
    r"""Drop the redundant trailing coordinate of an embedded LieGroup jacobian.

    Canonical embedded-to-tangent column map shared by all public PyPose
    differentiation APIs. Ordinary Tensors and Lie-algebra values are
    returned unchanged.
    """
    if isinstance(jacobian, (tuple, list)):
        return type(jacobian)(_slice_lie_jacobian(j, value) for j in jacobian)
    if isinstance(value, LieTensor) and not value.ltype.on_manifold:
        return jacobian[..., :int(value.ltype.manifold[0])]
    return jacobian


def _tangent_numel(value):
    r"""Flattened coordinate count of ``value`` after tangent-axis slicing.

    ``(numel // embedding) * manifold`` for an embedded group (e.g. SE3
    (N, 7) -> 6N), otherwise ``value.numel()``. Used by flatten paths so a
    sliced jacobian reshapes to the tangent width.
    """
    if isinstance(value, LieTensor) and not value.ltype.on_manifold:
        embedding = int(value.ltype.embedding[0])
        manifold = int(value.ltype.manifold[0])
        nelem = value.numel() // embedding
        return nelem * manifold
    return value.numel()


def jacobian(func: Callable, inputs, *, create_graph=False, strict=False,
             vectorize=False, strategy='reverse-mode'):
    r"""
    This function provides the exact same functionality as `torch.autograd.functional.jacobian()
    <https://pytorch.org/docs/stable/generated/torch.autograd.functional.jacobian.html>`_,
    except that it allows LieTensor to be used as input when calculating the jacobian.

    Args:
        func (function): a Python function that takes Tensor inputs and returns
            a Tensor or a tuple of Tensors.
        inputs (tuple of Tensors or Tensor): the inputs to the function ``func``.
        create_graph (bool, optional): if ``True``, the Jacobian will be computed
            in a differentiable manner. Default: ``False``.
        strict (bool, optional): if ``True``, an error will be raised when the
            output is independent of the input. Default: ``False``.
        vectorize (bool, optional): experimentally vectorizes the Jacobian
            computation. Default: ``False``.
        strategy (str, optional): either ``"forward-mode"`` or ``"reverse-mode"``.
            Default: ``"reverse-mode"``.

    Returns:
        Jacobian of ``func`` with respect to ``inputs``. For an embedded LieGroup
        input (including a LieGroup ``Parameter``, which keeps its embedded
        storage), the corresponding axis is in tangent coordinates rather than
        the embedded ones, e.g. SE3 contributes 6 columns, not 7.

    Examples:
        >>> import pypose as pp
        >>> import torch
        >>> def func(pose, points):
        ...     return pose @ points
        >>> pose = pp.randn_SE3(1)
        >>> points = torch.randn(1, 3)
        >>> J = pp.func.jacobian(func, (pose, points))
        >>> J[0].shape
        torch.Size([1, 3, 1, 6])
    """
    result = torch_jacobian(func, inputs, create_graph=create_graph, strict=strict,
                            vectorize=vectorize, strategy=strategy)
    if isinstance(inputs, (tuple, list)):
        return type(inputs)(_slice_lie_jacobian(j, value)
                            for j, value in zip(result, inputs))
    return _slice_lie_jacobian(result, inputs)


def jacrev(func: Callable, argnums: Union[int, Tuple[int]] = 0, *, has_aux=False,
           chunk_size: Optional[int] = None,
           _preallocate_and_copy=False):
    r"""
    This function provides the exact same functionality as `torch.func.jacrev()
    <https://pytorch.org/docs/stable/generated/torch.func.jacrev.html#torch.func.jacrev>`_,
    except that it allows LieTensor to be used as input when calculating the jacobian.

    Args:
        func (function): A Python function that takes one or more arguments,
            one of which must be a Tensor, and returns one or more Tensors
        argnums (int or Tuple[int]): Optional, integer or tuple of integers,
            saying which arguments to get the Jacobian with respect to.
            Default: 0.
        has_aux (bool): Flag indicating that ``func`` returns a
            ``(output, aux)`` tuple where the first element is the output of
            the function to be differentiated and the second element is
            auxiliary objects that will not be differentiated.
            Default: False.
        chunk_size (None or int): If None (default), use the maximum chunk size
            (equivalent to doing a single vmap over vjp to compute the jacobian).
            If 1, then compute the jacobian row-by-row with a for-loop.
            If not None, then compute the jacobian :attr:`chunk_size` rows at a time
            (equivalent to doing multiple vmap over vjp). If you run into memory issues
            computing the jacobian, please try to specify a non-None chunk_size.

    Returns:
        Returns a function that takes in the same inputs as ``func`` and
        returns the Jacobian of ``func`` with respect to the arg(s) at
        ``argnums``. If ``has_aux is True``, then the returned function
        instead returns a ``(jacobian, aux)`` tuple where ``jacobian``
        is the Jacobian and ``aux`` is auxiliary objects returned by ``func``.

    A basic usage with our LieTensor type would be the transformation function.

        >>> import pypose as pp
        >>> import torch
        >>> def func(pose, points):
        ...     return pose @ points
        >>> pose = pp.randn_SE3(1)
        >>> points = torch.randn(1, 3)
        >>> jacobian = pp.func.jacrev(func)(pose, points)
        >>> jacobian
        tensor([[[[ 1.0000,  0.0000,  0.0000,  0.0000,  1.5874, -0.2061]],
                [[ 0.0000,  1.0000,  0.0000, -1.5874,  0.0000, -1.4273]],
                [[ 0.0000,  0.0000,  1.0000,  0.2061,  1.4273,  0.0000]]]])
    """
    jac_func = torch.func.jacrev(func, argnums, has_aux=has_aux, chunk_size=chunk_size,
        _preallocate_and_copy=_preallocate_and_copy)
    @retain_ltype()
    def wrapper_fn(*args, **kwargs):
        result = jac_func(*args, **kwargs)
        if has_aux:
            result, aux = result
        if isinstance(argnums, tuple):
            result = tuple(_slice_lie_jacobian(result_i, args[i])
                           for result_i, i in zip(result, argnums))
        else:
            result = _slice_lie_jacobian(result, args[argnums])
        return (result, aux) if has_aux else result
    return wrapper_fn
