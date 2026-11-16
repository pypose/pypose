import pytest
import torch
import pypose as pp


@pytest.mark.parametrize('with_scale', [False, True])
@pytest.mark.parametrize('device,default_device', [
    ('cpu', 'cpu'),
    ('cpu', 'meta'),
    pytest.param('cuda', 'cpu', marks=pytest.mark.skipif(
        not torch.cuda.is_available(), reason='CUDA is not available')),
])
def test_svdstf_uses_input_device(device, default_device, with_scale):
    source = torch.tensor([[3., 0., 0.], [-3., 0., 0.],
                           [0., 2., 0.], [0., -2., 0.],
                           [0., 0., 1.], [0., 0., -1.]],
                          dtype=torch.float64, device=device)
    rotation = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]],
                            dtype=source.dtype, device=device)
    translation = torch.tensor([1., 2., 3.], dtype=source.dtype, device=device)
    scale = 2. if with_scale else 1.
    target = scale * source @ rotation.mT + translation

    # A default device must not redirect temporaries away from explicit inputs.
    with torch.device(default_device):
        transform = pp.svdstf(source, target, with_scale=with_scale)

    assert transform.device == source.device
    assert transform.dtype == source.dtype
    torch.testing.assert_close(transform.Act(source), target)
    torch.testing.assert_close(transform.translation(), translation)
