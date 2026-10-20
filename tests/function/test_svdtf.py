import pytest
import torch
import pypose as pp


def axis_points(dtype):
    return torch.tensor([[3., 0., 0.], [-3., 0., 0.],
                         [0., 2., 0.], [0., -2., 0.],
                         [0., 0., 1.], [0., 0., -1.]], dtype=dtype)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('axis', [0, 1, 2])
def test_svdtf_reflection_minimizes_rigid_error(dtype, axis):
    source = axis_points(dtype)
    signs = torch.ones(3, dtype=dtype)
    signs[axis] = -1
    target = source * signs

    # The z direction has the least variance. Keeping the two stronger axes
    # aligned gives the unique proper-rotation optimum, with total error 8.
    proper_signs = signs * torch.tensor([1., 1., -1.], dtype=dtype)
    expected = torch.diag(proper_signs)
    transform = pp.svdtf(source, target)
    torch.testing.assert_close(transform.rotation().matrix(), expected)
    torch.testing.assert_close(transform.rotation().matrix().det(),
                               torch.tensor(1., dtype=dtype))
    torch.testing.assert_close((transform.Act(source) - target).square().sum(),
                               torch.tensor(8., dtype=dtype))


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_svdtf_exact_planar_rigid_alignment(dtype):
    source = axis_points(dtype)[:4]
    rotation = torch.diag(torch.tensor([-1., 1., -1.], dtype=dtype))
    translation = torch.tensor([0.4, -0.7, 1.2], dtype=dtype)
    target = source @ rotation.mT + translation
    transform = pp.svdtf(source, target)
    torch.testing.assert_close(transform.Act(source), target)
    torch.testing.assert_close(transform.rotation().matrix(), rotation)
    torch.testing.assert_close(transform.translation(), translation)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_svdtf_exact_full_rank_rigid_alignment(dtype):
    source = axis_points(dtype)
    rotation = torch.tensor([[0., -1., 0.], [1., 0., 0.],
                             [0., 0., 1.]], dtype=dtype)
    translation = torch.tensor([0.4, -0.7, 1.2], dtype=dtype)
    target = source @ rotation.mT + translation
    transform = pp.svdtf(source, target)
    torch.testing.assert_close(transform.Act(source), target)
    torch.testing.assert_close(transform.rotation().matrix(), rotation)
    torch.testing.assert_close(transform.translation(), translation)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('source_batch,target_batch',
                         [((), ()), ((2,), (2,)), ((2, 1), (1, 3))])
def test_svdtf_reflection_batch_and_frames(dtype, source_batch, target_batch):
    points = axis_points(dtype)
    source_rotation = torch.tensor([[0., -1., 0.], [1., 0., 0.],
                                    [0., 0., 1.]], dtype=dtype)
    target_rotation = torch.tensor([[1., 0., 0.], [0., 0., -1.],
                                    [0., 1., 0.]], dtype=dtype)
    source_center = torch.ones(source_batch + (3,), dtype=dtype) * 0.3
    target_center = torch.ones(target_batch + (3,), dtype=dtype) * -0.7
    signs = torch.ones(target_batch + (3,), dtype=dtype)
    signs[..., 0] = torch.arange(signs[..., 0].numel()).reshape(target_batch) % 2 * 2 - 1
    source = points @ source_rotation.mT + source_center.unsqueeze(-2)
    target = (points * signs.unsqueeze(-2)) @ target_rotation.mT
    target = target + target_center.unsqueeze(-2)

    # An independent axis-aligned optimum, expressed in the two input frames.
    proper_signs = signs.clone()
    proper_signs[..., -1] *= signs.prod(dim=-1)
    expected_rotation = target_rotation @ torch.diag_embed(proper_signs) @ source_rotation.mT
    expected_translation = target_center - pp.bmv(expected_rotation, source_center)
    batch = torch.broadcast_shapes(source_batch, target_batch)
    transform = pp.svdtf(source, target)
    assert transform.shape == batch + (7,)
    torch.testing.assert_close(transform.rotation().matrix(),
                               expected_rotation.expand(batch + (3, 3)))
    torch.testing.assert_close(transform.translation(), expected_translation)
    expected_points = pp.bmv(expected_rotation.unsqueeze(-3), source) + expected_translation.unsqueeze(-2)
    torch.testing.assert_close(transform.unsqueeze(-2).Act(source), expected_points)


@pytest.mark.parametrize('reflected', [False, True])
def test_svdtf_output_gradcheck(reflected):
    source = axis_points(torch.float64)
    signs = torch.tensor([1., 1., -1. if reflected else 1.], dtype=source.dtype)
    target = source * signs
    target = target + torch.tensor([[0.1, 0.2, 0.0], [0.0, 0.1, -0.1],
                                    [0.2, 0.0, 0.1], [0.1, -0.1, 0.0],
                                    [-0.1, 0.2, 0.1], [0.0, -0.2, 0.2]], dtype=source.dtype)
    source.requires_grad_()
    target.requires_grad_()
    assert torch.autograd.gradcheck(lambda a, b: pp.svdtf(a, b).tensor(),
                                    (source, target))
