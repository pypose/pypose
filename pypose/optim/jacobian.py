import sys
import math
import copy
import torch
import warnings
from torch import nn, Tensor
from torch.autograd.functional import jacobian
from typing import List, Tuple, Dict, Union, Callable


# Utilities to make nn.Module "functional"
# In particular the goal is to be able to provide a function that takes as input
# the parameters and evaluate the nn.Module using fixed inputs.
def _del_nested_attr(obj: nn.Module, names: List[str]) -> None:
    """
    Deletes the attribute specified by the given list of names.
    For example, to delete the attribute obj.conv.weight,
    use _del_nested_attr(obj, ['conv', 'weight'])
    """
    if len(names) == 1:
        delattr(obj, names[0])
    else:
        _del_nested_attr(getattr(obj, names[0]), names[1:])


def _set_nested_attr(obj: nn.Module, names: List[str], value: Tensor) -> None:
    """
    Set the attribute specified by the given list of names to value.
    For example, to set the attribute obj.conv.weight,
    use _del_nested_attr(obj, ['conv', 'weight'], value)
    """
    if len(names) == 1:
        setattr(obj, names[0], value)
    else:
        _set_nested_attr(getattr(obj, names[0]), names[1:], value)


def extract_weights(mod: nn.Module) -> Tuple[Tuple[Tensor, ...], List[str]]:
    """
    This function removes all the Parameters from the model and
    return them as a tuple as well as their original attribute names.
    The weights must be re-loaded with `load_weights` before the model
    can be used again.
    Note that this function modifies the model in place and after this
    call, mod.parameters() will be empty.
    """
    orig_params = tuple(mod.parameters())
    # Remove all the parameters in the model
    names = []
    for name, p in list(mod.named_parameters()):
        _del_nested_attr(mod, name.split("."))
        names.append(name)

    # Make params regular Tensors instead of nn.Parameter
    params = tuple(p.detach().requires_grad_() for p in orig_params)
    return params, names


def load_weights(mod: nn.Module, names: List[str], params: Tuple[Tensor, ...]) -> None:
    """
    Reload a set of weights so that `mod` can be used again to perform a forward pass.
    Note that the `params` are regular Tensors (that can have history) and so are left
    as Tensors. This means that mod.parameters() will still be empty after this call.
    """
    for name, p in zip(names, params):
        _set_nested_attr(mod, name.split("."), p)


def module_jacobian(module, inputs, create_graph=False, strict=False, vectorize=False, strategy='reverse-mode'):
    r'''
    Compute the Jacobian of module output with respect to module parameters.

    Args:
        module (torch.nn.Module): a PyTorch module that takes Tensor inputs and
            returns a tuple of Tensors or a Tensor.
        inputs (tuple of Tensors or Tensor): inputs to the function ``func``.
        create_graph (bool, optional): If ``True``, the Jacobian will be
            computed in a differentiable manner. Note that when ``strict`` is
            ``False``, the result can not require gradients or be disconnected
            from the inputs.  Defaults to ``False``.
        strict (bool, optional): If ``True``, an error will be raised when we
            detect that there exists an input such that all the outputs are
            independent of it. If ``False``, we return a Tensor of zeros as the
            jacobian for said inputs, which is the expected mathematical value.
            Defaults to ``False``.
        vectorize (bool, optional): When computing the jacobian, usually we invoke
            ``autograd.grad`` once per row of the jacobian. If this flag is
            ``True``, we perform only a single ``autograd.grad`` call with
            ``batched_grad=True`` which uses the vmap prototype feature.
            Though this should lead to performance improvements in many cases,
            because this feature is still experimental, there may be performance
            cliffs. See :func:`torch.autograd.grad`'s ``batched_grad`` parameter for
            more information.
        strategy (str, optional): Set to ``"forward-mode"`` or ``"reverse-mode"`` to
            determine whether the Jacobian will be computed with forward or reverse
            mode AD. Currently, ``"forward-mode"`` requires ``vectorized=True``.
            Defaults to ``"reverse-mode"``. If ``func`` has more outputs than
            inputs, ``"forward-mode"`` tends to be more performant. Otherwise,
            prefer to use ``"reverse-mode"``.

    Returns:
        Jacobian (Tensor or nested tuple of Tensors): if there is a single
        input and output, this will be a single Tensor containing the
        Jacobian for the linearized inputs and output. If one of the two is
        a tuple, then the Jacobian will be a tuple of Tensors. If both of
        them are tuples, then the Jacobian will be a tuple of tuple of
        Tensors where ``Jacobian[i][j]`` will contain the Jacobian of the
        ``i``\th output and ``j``\th input and will have as size the
        concatenation of the sizes of the corresponding output and the
        corresponding input and will have same dtype and device as the
        corresponding input. If strategy is ``forward-mode``, the dtype will be
        that of the output; otherwise, the input.

    Note:
        Multiple module parameters are flattened to support parallel computing,
        therefore the last dimension of Jacobian is the number of total parameters
        in the module.

    Warning:
        This function is in contrast to PyTorch's function `jacobian
        <https://pytorch.org/docs/stable/generated/torch.autograd.functional.jacobian.html>`_,
        which computes the Jacobian of a given Python function.

    Example:
        >>> inputs = torch.randn(2, 2, 2)
        >>> module = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=1)
        >>> J = pp.optim.module_jacobian(module, inputs)
        tensor([[[[-1.1571, -1.6217,  0.0000,  0.0000,  1.0000,  0.0000],
                  [ 0.2917, -1.1545,  0.0000,  0.0000,  1.0000,  0.0000]],
                [[-1.4052,  0.7642,  0.0000,  0.0000,  1.0000,  0.0000],
                 [ 0.7777, -1.5251,  0.0000,  0.0000,  1.0000,  0.0000]]],
                [[[ 0.0000,  0.0000, -1.1571, -1.6217,  0.0000,  1.0000],
                  [ 0.0000,  0.0000,  0.2917, -1.1545,  0.0000,  1.0000]],
                [[ 0.0000,  0.0000, -1.4052,  0.7642,  0.0000,  1.0000],
                 [ 0.0000,  0.0000,  0.7777, -1.5251,  0.0000,  1.0000]]]])
        >>> J.shape
        torch.Size([2, 2, 2, 6])
    '''
    params, names = extract_weights(module) # deparameterize weights
    numels, shapes, params = zip(*[(p.numel(), p.shape, p.view(-1)) for p in params])
    param = torch.cat(params, dim=-1)

    def param_as_input(param):
        param = torch.split(param, numels)
        params = [p.view(s) for p, s in zip(param, shapes)]
        load_weights(module, names, params)
        return module(inputs)

    return jacobian(param_as_input, param, create_graph, strict, vectorize, strategy)
