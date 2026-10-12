import copy
import pytest
import torch
import pypose as pp


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA unavailable'))])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_warm_imu_state_migration(device, dtype):
    dt = torch.full((1, 4, 1), 0.01)
    gyro = torch.tensor([0.1, -0.2, 0.3]).expand(1, 4, 3)
    acc = torch.tensor([0.2, 0.1, 9.7]).expand(1, 4, 3)
    module = pp.module.IMUPreintegrator()
    module(dt, gyro, acc)
    reference = copy.deepcopy(module)
    reference.Rij = reference.Rij.to(device=device, dtype=dtype)
    reference.to(device=device, dtype=dtype)
    module.to(device=device, dtype=dtype)
    assert module.Rij.device == module.pos.device
    assert module.Rij.dtype == dtype
    assert isinstance(module.Rij, pp.LieTensor)
    assert module.state_dict() == {}
    args = [value.to(device=device, dtype=dtype) for value in (dt, gyro, acc)]
    output, expected = module(*args), reference(*args)
    for key in output:
        torch.testing.assert_close(output[key], expected[key])
    module.cpu().float()
    assert module.Rij.device.type == 'cpu' and module.Rij.dtype == torch.float32
    assert module.state_dict() == {}
    module(dt, gyro, acc)


def test_cold_and_reset_cache_remains_nonpersistent():
    for reset in [False, True]:
        module = pp.module.IMUPreintegrator(reset=reset).double()
        assert module.Rij is None and module.state_dict() == {}
        module(torch.ones(1, 2, 1, dtype=torch.float64) * 0.01,
               torch.zeros(1, 2, 3, dtype=torch.float64),
               torch.zeros(1, 2, 3, dtype=torch.float64))
        assert module.state_dict() == {}
        if reset:
            assert module.Rij is None
