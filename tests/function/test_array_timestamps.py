import warnings

import numpy as np
import pytest
import torch
import pypose as pp
from pypose.metric.ape_rpe import StampedSE3


def as_input(values, kind):
    if kind == 'list':
        return values.tolist()
    if kind == 'tuple':
        return tuple(values)
    if kind == 'numpy_strided':
        storage = np.repeat(values, 2)
        return storage[::2]
    if kind == 'numpy_readonly':
        result = values.copy()
        result.flags.writeable = False
        return result
    if kind.startswith('numpy_'):
        return values.astype(kind.removeprefix('numpy_'))
    if kind.startswith('tensor_'):
        return torch.tensor(values, dtype=getattr(torch, kind.removeprefix('tensor_')))
    return None


def make_poses():
    reference = pp.identity_SE3(4, dtype=torch.float64)
    reference[:, 0] = torch.arange(4, dtype=torch.float64)
    estimate = reference.clone()
    estimate.tensor()[:, 0] += torch.tensor([0.2, 0.4, 0.7, 1.1], dtype=torch.float64)
    return reference, estimate


@pytest.mark.parametrize('metric,expected', (('ape', 0.6), ('rpe', 0.3)))
@pytest.mark.parametrize('kind', ('list', 'tuple', 'numpy_float32', 'numpy_float64',
                                 'numpy_int64', 'numpy_strided', 'numpy_readonly',
                                 'tensor_float32', 'tensor_float64', 'none'))
def test_public_metrics_accept_documented_timestamp_inputs(metric, expected, kind):
    reference, estimate = make_poses()
    rstamp = as_input(np.arange(4, dtype=np.float64), kind)
    estamp = as_input(np.arange(4, dtype=np.float64), kind)
    original = None if rstamp is None else np.asarray(rstamp).copy()
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        result = getattr(pp.metric, metric)(rstamp, reference, estamp, estimate, otype='Mean')
    torch.testing.assert_close(result, result.new_tensor(expected))
    if original is not None:
        np.testing.assert_array_equal(np.asarray(rstamp), original)
        np.testing.assert_array_equal(np.asarray(estamp), original)


@pytest.mark.parametrize('metric,expected', (('ape', 0.6), ('rpe', 0.3)))
@pytest.mark.parametrize('kind', ('list', 'tuple', 'numpy_strided', 'numpy_readonly'))
def test_fractional_epoch_timestamps_keep_precision_and_external_storage(metric, expected, kind):
    reference, estimate = make_poses()
    values = 1_700_000_000 + np.arange(4, dtype=np.float64) * 0.01
    rstamp, estamp = as_input(values, kind), as_input(values - 0.125, kind)
    before_reference, before_estimate = np.asarray(rstamp).copy(), np.asarray(estamp).copy()
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        result = getattr(pp.metric, metric)(rstamp, reference, estamp, estimate,
                                            offset=0.125, diff=0.001, otype='Mean')
    torch.testing.assert_close(result, result.new_tensor(expected))
    np.testing.assert_array_equal(np.asarray(rstamp), before_reference)
    np.testing.assert_array_equal(np.asarray(estamp), before_estimate)


@pytest.mark.parametrize('device', ('cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA is unavailable'))))
def test_array_conversion_matches_pose_device(device):
    poses = pp.identity_SE3(4, dtype=torch.float32, device=device)
    result = StampedSE3([0, 1, 2, 3], poses)
    assert result.timestamps.device == poses.device
    assert result.timestamps.dtype == torch.float64
    torch.testing.assert_close(result.timestamps.cpu(), torch.arange(4, dtype=torch.float64))


@pytest.mark.parametrize('dtype', (torch.float32, torch.float64))
def test_existing_tensor_conversion_keeps_gradient_and_storage_contract(dtype):
    stamps = torch.arange(4, dtype=dtype, requires_grad=True)
    result = StampedSE3(stamps, pp.identity_SE3(4, dtype=dtype))
    result.timestamps.square().sum().backward()
    torch.testing.assert_close(stamps.grad, 2 * stamps.detach())
    if dtype == torch.float64:
        assert result.timestamps.data_ptr() == stamps.data_ptr()


@pytest.mark.parametrize('stamps,message', (
    ([0, 1], 'same length'),
    ([[0, 1, 2, 3]], 'one array'),
    ([0, 2, 1, 3], 'accending'),
))
def test_array_inputs_keep_timestamp_validation(stamps, message):
    with pytest.raises(AssertionError, match=message):
        StampedSE3(stamps, pp.identity_SE3(4))
