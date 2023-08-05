from typing import Callable, Union, Tuple, List, Any, Optional
import torch
from functools import partial, wraps
from torch._functorch.vmap import vmap, doesnt_support_saved_tensors_hooks, get_chunk_sizes
from torch._functorch.eager_transforms import error_if_complex, _slice_argnums, \
    _chunked_standard_basis_for_, _safe_zero_index, _vjp_with_argnums
from torch.utils._pytree import tree_flatten, tree_unflatten, tree_map, tree_map_only

from pypose.lietensor.lietensor import retain_ltype


def jacrev(func: Callable, argnums: Union[int, Tuple[int]] = 0, *, has_aux=False,
           chunk_size: Optional[int] = None,
           _preallocate_and_copy=False):
    """
    This function provides the exact same functionality as `torch.func.jacrev`, except
    that it allows LieTensor to be used as input.
    """
    jac_func = torch.func.jacrev(func, argnums, has_aux=has_aux, chunk_size=chunk_size,
        _preallocate_and_copy=_preallocate_and_copy)
    @retain_ltype()
    def wrapper_fn(*args, **kwargs):
        return jac_func(*args, **kwargs)
    return wrapper_fn
