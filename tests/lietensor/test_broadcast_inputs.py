import pytest
import torch
import pypose as pp

from pypose.lietensor.operation import broadcast_inputs


DEVICES = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])
GROUPS = [('SO3', 3), ('SE3', 6), ('RxSO3', 4), ('Sim3', 7)]


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('shapes', [((1,), (19,)), ((2, 1), (1, 3)), ((), ())])
def test_broadcast_inputs_preserves_storage(device, shapes):
    x = torch.randn(*shapes[0], 7, device=device)
    y = torch.randn(*shapes[1], 3, device=device)
    (bx, by), batch = broadcast_inputs(x, y)
    assert batch == torch.broadcast_shapes(x.shape[:-1], y.shape[:-1])
    assert bx.untyped_storage().data_ptr() == x.untyped_storage().data_ptr()
    assert by.untyped_storage().data_ptr() == y.untyped_storage().data_ptr()
    # Kernels accept the original batch dimensions; flattening a Cartesian
    # broadcast would materialize a copy even without an explicit contiguous().
    expected_batch = batch or (1,)
    assert bx.shape == expected_batch + (7,)
    assert by.shape == expected_batch + (3,)
    torch.testing.assert_close(bx, x.expand_as(bx))
    torch.testing.assert_close(by, y.expand_as(by))


@pytest.mark.parametrize('device', DEVICES)
def test_broadcast_inputs_unary_noncontiguous(device):
    x = torch.randn(3, 2, 7, device=device).transpose(0, 1)
    (out,), batch = broadcast_inputs(x, None)
    assert batch == (2, 3)
    assert out.untyped_storage().data_ptr() == x.untyped_storage().data_ptr()
    torch.testing.assert_close(out, x)


def apply_operation(group, operation, x, y):
    X = group(x)
    if operation == 'Mul':
        return (X * group(y)).tensor()
    out = X.Act(y) if operation.startswith('Act') else getattr(X, operation)(y)
    return out.tensor() if isinstance(out, pp.LieTensor) else out


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('group,dimension', GROUPS)
@pytest.mark.parametrize('operation', ['Act3', 'Act4', 'Mul', 'Adj', 'AdjT', 'Jinvp'])
@pytest.mark.parametrize('layout', ['cartesian', 'noncontiguous', 'unbatched', 'empty'])
def test_broadcast_operations_match_flattened_copy(device, dtype, group, dimension,
                                                  operation, layout):
    torch.manual_seed(252)
    kwargs = dict(device=device, dtype=dtype)
    if layout == 'cartesian':
        xs, ys = (2, 1), (1, 3)
    elif layout == 'noncontiguous':
        xs, ys = (3, 2), (3, 2)
    elif layout == 'empty':
        xs, ys = (0, 1), (1, 3)
    else:
        xs = ys = ()
    randn = getattr(pp, 'randn_' + group)
    group = getattr(pp, group)
    x = randn(*xs, sigma=0.2, **kwargs).tensor().detach()
    if operation == 'Mul':
        y = randn(*ys, sigma=0.2, **kwargs).tensor().detach()
    else:
        width = int(operation[-1]) if operation.startswith('Act') else dimension
        y = torch.randn(*ys, width, **kwargs)
    if layout == 'noncontiguous':
        x, y = x.transpose(0, 1), y.transpose(0, 1)
        assert not x.is_contiguous() and not y.is_contiguous()

    x, y = x.requires_grad_(), y.requires_grad_()
    actual = apply_operation(group, operation, x, y)
    weight = torch.randn_like(actual)
    actual_grads = torch.autograd.grad((actual * weight).sum(), (x, y))

    # The previous implementation expanded, flattened, and copied both inputs.
    # Reproduce that path explicitly, keeping its graph back to the small inputs
    # so that gradient reductions across broadcast dimensions are checked too.
    xr, yr = x.detach().requires_grad_(), y.detach().requires_grad_()
    batch = torch.broadcast_shapes(xr.shape[:-1], yr.shape[:-1])
    xf = xr.expand(batch + (xr.shape[-1],)).reshape(-1, xr.shape[-1]).contiguous()
    yf = yr.expand(batch + (yr.shape[-1],)).reshape(-1, yr.shape[-1]).contiguous()
    expected = apply_operation(group, operation, xf, yf).reshape(actual.shape)
    expected_grads = torch.autograd.grad((expected * weight).sum(), (xr, yr))
    torch.testing.assert_close(actual, expected)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad)
