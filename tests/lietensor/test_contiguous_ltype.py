import pytest
import torch
import pypose as pp


KINDS = ('SO3', 'SE3', 'Sim3', 'RxSO3', 'so3', 'se3', 'sim3', 'rxso3')


@pytest.mark.parametrize('kind', KINDS)
@pytest.mark.parametrize('strided', (False, True))
@pytest.mark.parametrize('dtype', (torch.float32, torch.float64))
def test_contiguous_keeps_lie_type_values_and_copy_contract(kind, strided, dtype):
    source = getattr(pp, 'randn_' + kind)(6, dtype=dtype)
    view = source[::2] if strided else source
    result = view.contiguous()
    assert isinstance(result, pp.LieTensor)
    assert result.ltype is view.ltype
    assert result.lshape == view.lshape
    assert result.is_contiguous()
    torch.testing.assert_close(result.tensor(), view.tensor())
    assert (result.data_ptr() == view.data_ptr()) is (not strided)
    if kind[0].isupper():
        torch.testing.assert_close(result.Log(), view.Log())
    else:
        torch.testing.assert_close(result.Exp(), view.Exp())


@pytest.mark.parametrize('kind', KINDS)
def test_contiguous_preserves_gradient_from_strided_inputs(kind):
    source = getattr(pp, 'randn_' + kind)(6, dtype=torch.float64, requires_grad=True)
    result = source[::2].contiguous()
    assert isinstance(result, pp.LieTensor)
    actual, = torch.autograd.grad(result.tensor().square().sum(), source)
    reference, = torch.autograd.grad(source.tensor()[::2].clone().square().sum(), source)
    torch.testing.assert_close(actual, reference)


@pytest.mark.parametrize('kind', KINDS)
def test_empty_contiguous_batches_keep_domain_operations(kind):
    source = getattr(pp, 'randn_' + kind)(0, 3, dtype=torch.float64)
    result = source.contiguous()
    assert isinstance(result, pp.LieTensor)
    assert result.ltype is source.ltype
    assert result.lshape == (0, 3)
    transformed = result.Log() if kind[0].isupper() else result.Exp()
    assert transformed.lshape == (0, 3)


@pytest.mark.parametrize('kind', KINDS)
def test_channel_last_storage_keeps_lie_coordinates(kind):
    source = getattr(pp, 'randn_' + kind)(2, 3, 4, dtype=torch.float64)
    result = source.contiguous(memory_format=torch.channels_last)
    assert isinstance(result, pp.LieTensor)
    assert result.ltype is source.ltype
    assert result.is_contiguous(memory_format=torch.channels_last)
    torch.testing.assert_close(result.tensor(), source.tensor())
    if kind[0].isupper():
        torch.testing.assert_close(result.Log(), source.Log())
    else:
        torch.testing.assert_close(result.Exp(), source.Exp())


def test_prepared_pose_batch_remains_usable_in_public_bspline():
    source = pp.randn_SE3(16, dtype=torch.float64)[::2]
    result = pp.bspline(source.contiguous(), interval=0.3)
    expected = pp.bspline(source, interval=0.3)
    torch.testing.assert_close(result, expected)


@pytest.mark.parametrize('dtype', (torch.float32, torch.float64))
def test_contiguous_parameter_keeps_lie_type_and_gradient(dtype):
    parameter = pp.Parameter(pp.randn_se3(4, dtype=dtype))
    result = parameter.contiguous()
    assert isinstance(result, pp.LieTensor)
    assert result.ltype is parameter.ltype
    gradient, = torch.autograd.grad(result.tensor().sum(), parameter)
    torch.testing.assert_close(gradient, torch.ones_like(parameter.tensor()))


@pytest.mark.parametrize('strided', (False, True))
def test_ordinary_tensor_contiguous_remains_an_ordinary_tensor(strided):
    source = torch.arange(12).reshape(4, 3)
    result = (source[::2] if strided else source).contiguous()
    assert type(result) is torch.Tensor
