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
    that it
    """
    if not (chunk_size is None or chunk_size > 0):
        raise ValueError("jacrev: `chunk_size` should be greater than 0.")

    @retain_ltype()
    @wraps(func)
    def wrapper_fn(*args):
        error_if_complex("jacrev", args, is_input=True)
        vjp_out = _vjp_with_argnums(func, *args, argnums=argnums, has_aux=has_aux)
        if has_aux:
            output, vjp_fn, aux = vjp_out
        else:
            output, vjp_fn = vjp_out

        # See NOTE: [Computing jacobian with vmap and vjp for multiple outputs]
        flat_output, output_spec = tree_flatten(output)

        error_if_complex("jacrev", flat_output, is_input=False)

        # NB: vjp already checks that all outputs are tensors
        # Step 1: Construct grad_outputs by splitting the standard basis
        flat_output_numels = tuple(out.numel() for out in flat_output)

        primals = _slice_argnums(args, argnums)
        flat_primals, primals_spec = tree_flatten(primals)

        def compute_jacobian_stacked():
            # Helper function to compute chunked Jacobian
            # The intermediate chunked calculation are only
            # scoped at this function level.
            chunked_results = []
            for flat_basis_chunk in _chunked_standard_basis_for_(flat_output,
                                                                 flat_output_numels,
                                                                 chunk_size=chunk_size):
                if chunk_size == 1:
                    # sanity check.
                    for t in flat_basis_chunk:
                        assert t.size(0) == 1

                    flat_basis_chunk = tree_map(lambda t: torch.squeeze(t, 0), flat_basis_chunk)

                basis = tree_unflatten(flat_basis_chunk, output_spec)

                if chunk_size == 1:
                    # Behaviour with `chunk_size=1` is same as `for-loop`
                    # i.e. user shouldn't deal with the limitations of vmap.
                    chunked_result = vjp_fn(basis)
                else:  # chunk_size is None or chunk_size != 1
                    chunked_result = vmap(vjp_fn)(basis)

                flat_results, _ = tree_flatten(chunked_result)

                if chunk_size == 1:
                    flat_results = tree_map(lambda t: torch.unsqueeze(t, 0), flat_results)

                chunked_results.append(flat_results)

            if len(chunked_results) == 1:
                # Short-circuit if we used a single chunk
                return chunked_results[0]

            # Concatenate chunks.
            flat_results = []
            # Iterate and concat the jacobians of different
            # inputs.
            for idx in range(len(flat_primals)):
                r = tuple(map(lambda r_: r_[idx], chunked_results))
                flat_results.append(torch.cat(r, 0))

            return flat_results

        def compute_jacobian_preallocate_and_copy():
            # Helper function to compute chunked Jacobian
            # The intermediate chunked calculation are only
            # scoped at this function level.
            out_vec_size = sum(flat_output_numels)

            # Don't pre-allocate if we have a single chunk.
            if not (chunk_size is None or chunk_size >= out_vec_size):
                stacked_results = [primal.new_zeros(out_vec_size, *primal.shape) for primal in flat_primals]

            for idx, flat_basis_chunk in enumerate(_chunked_standard_basis_for_(flat_output,
                                                                                flat_output_numels,
                                                                                chunk_size=chunk_size)):
                if chunk_size == 1:
                    # sanity check.
                    for t in flat_basis_chunk:
                        assert t.size(0) == 1

                    flat_basis_chunk = list(map(lambda t: torch.squeeze(t, 0), flat_basis_chunk))

                basis = tree_unflatten(flat_basis_chunk, output_spec)

                if chunk_size == 1:
                    # Behaviour with `chunk_size=1` is same as `for-loop`
                    # i.e. user shouldn't deal with the limitations of vmap.
                    chunked_result = vjp_fn(basis)
                else:  # chunk_size is None or chunk_size != 1
                    chunked_result = vmap(vjp_fn)(basis)

                flat_results, _ = tree_flatten(chunked_result)

                # Short-circuit if we have a single chunk.
                if chunk_size is None or chunk_size >= out_vec_size:
                    if chunk_size == 1:  # and out_vec_size == 1
                        # Since we squeezed the output dim
                        flat_results = tree_map(lambda t: torch.unsqueeze(t, 0), flat_results)
                    return flat_results

                for r, sr in zip(flat_results, stacked_results):
                    sr[idx * chunk_size: (idx + 1) * chunk_size].copy_(r)

            return stacked_results

        if _preallocate_and_copy:
            flat_jacobians_per_input = compute_jacobian_preallocate_and_copy()
        else:
            flat_jacobians_per_input = compute_jacobian_stacked()

        # Step 2: The returned jacobian is one big tensor per input. In this step,
        # we split each Tensor by output.
        flat_jacobians_per_input = [result.split(flat_output_numels, dim=0) for result in flat_jacobians_per_input]
        flat_input_flat_output = [
            tuple(split.view(out.shape + primal.shape)
                  for split, out in zip(splits, flat_output))
            for splits, primal in zip(flat_jacobians_per_input, flat_primals)
        ]

        # Step 3: Right now, `jacobian` is a List[List[Tensor]].
        # The outer List corresponds to the number of primals,
        # the inner List corresponds to the number of outputs.
        # We need to:
        # a. Exchange the order of the outer List and inner List
        # b. tree_unflatten the inner Lists (which correspond to the primals)
        # c. handle the argnums=int case
        # d. tree_unflatten the outer List (which corresponds to the outputs)
        flat_output_flat_input = tuple(zip(*flat_input_flat_output))

        flat_output_input = tuple(tree_unflatten(flat_input, primals_spec)
                                  for flat_input in flat_output_flat_input)

        if isinstance(argnums, int):
            flat_output_input = tuple(_safe_zero_index(flat_input)
                                      for flat_input in flat_output_input)
        output_input = tree_unflatten(flat_output_input, output_spec)
        if has_aux:
            return output_input, aux
        return output_input
    return wrapper_fn
