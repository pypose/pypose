import pytest
import torch
import pypose as pp


@pytest.mark.parametrize('length', [0, 1, 3, 4, 5, 8, 9, 17])
@pytest.mark.parametrize('dim', [0, 1, -1, -2])
@pytest.mark.parametrize('inplace', [False, True])
def test_cumops_prefix_sum_lengths(length, dim, inplace):
    shape = (length, 3) if dim % 2 == 0 else (3, length)
    values = torch.arange(3 * length, dtype=torch.int64).reshape(shape)
    original = values.clone()
    operation = pp.cumops_ if inplace else pp.cumops

    actual = operation(values, dim, torch.add)

    torch.testing.assert_close(actual, original.cumsum(dim))
    if inplace:
        assert actual is values
    else:
        assert actual is not values
        torch.testing.assert_close(values, original)


def ordered_matrix_prefix(values, dim, left):
    if values.shape[dim] == 0:
        return values.clone()
    prefixes = []
    for value in values.unbind(dim):
        if not prefixes:
            product = value
        else:
            product = value @ product if left else product @ value
        prefixes.append(product)
    return torch.stack(prefixes, dim)


@pytest.mark.parametrize('length', [0, 1, 3, 5, 8])
@pytest.mark.parametrize('name', ['cumprod', 'cumprod_', 'cummul', 'cummul_'])
@pytest.mark.parametrize('left', [False, True])
@pytest.mark.parametrize('batched', [False, True])
def test_lie_prefix_product_lengths(length, name, left, batched):
    # Noncommuting right-angle rotations give an independent matrix-product oracle.
    rotations = torch.tensor([
        [[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]],
        [[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]],
        [[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]],
    ], dtype=torch.float64)
    rotations = rotations.repeat((length + 2) // 3, 1, 1)[:length]
    dim = 0
    if batched:
        rotations = torch.stack((rotations, rotations.flip(0)))
        dim = 1
    values = pp.mat2SO3(rotations)
    original = values.tensor().clone()
    expected = ordered_matrix_prefix(rotations, dim, left)

    actual = getattr(pp, name)(values, dim, left=left)

    assert isinstance(actual, pp.LieTensor)
    assert actual.ltype is pp.SO3_type
    torch.testing.assert_close(actual.matrix(), expected, rtol=1e-12, atol=1e-12)
    if name.endswith('_'):
        assert actual is values
    else:
        torch.testing.assert_close(values.tensor(), original)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('dim', [1, -1])
def test_noncontiguous_prefix_sum_gradient(dtype, dim):
    values = torch.linspace(-1., 1., 15, dtype=dtype).reshape(5, 3).T
    values.requires_grad_()
    assert not values.is_contiguous()
    weights = torch.arange(1, 16, dtype=dtype).reshape(3, 5)

    actual = pp.cumops(values, dim, torch.add)
    gradient, = torch.autograd.grad((actual * weights).sum(), values)

    torch.testing.assert_close(actual, values.cumsum(dim))
    expected_gradient = weights.flip((dim,)).cumsum(dim).flip((dim,))
    torch.testing.assert_close(gradient, expected_gradient)
