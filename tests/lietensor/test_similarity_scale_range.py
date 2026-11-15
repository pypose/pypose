import pytest
import torch
import pypose as pp


@pytest.mark.parametrize("convert", [pp.mat2Sim3, pp.mat2RxSO3])
@pytest.mark.parametrize("dtype, scales", [
    (torch.float32, [1e13, 1e20]),
    (torch.float64, [1e110, 1e150]),
])
@pytest.mark.parametrize("shape", [(3, 3), (3, 4), (4, 4)])
def test_similarity_conversion_avoids_determinant_overflow(convert, dtype, scales, shape):
    rotation = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]], dtype=dtype)
    scale = torch.tensor(scales, dtype=dtype)
    matrix = torch.zeros(2, *shape, dtype=dtype)
    matrix[..., :3, :3] = scale[:, None, None] * rotation
    if shape[-1] == 4:
        matrix[..., :3, 3] = torch.tensor([1., -2., 3.], dtype=dtype)
    if shape[-2] == 4:
        matrix[..., 3, 3] = 1

    assert torch.isinf(torch.det(matrix[..., :3, :3])).all()
    result = convert(matrix)
    assert torch.isfinite(result.tensor()).all()
    torch.testing.assert_close(result.scale().squeeze(-1), scale, rtol=1e-5, atol=0)
    torch.testing.assert_close(result.rotation().matrix(), rotation.expand(2, 3, 3))
    if convert is pp.mat2Sim3:
        expected = matrix[..., :3, 3] if shape[-1] == 4 else torch.zeros(2, 3, dtype=dtype)
        torch.testing.assert_close(result.translation(), expected)


@pytest.mark.parametrize("convert", [pp.mat2Sim3, pp.mat2RxSO3])
@pytest.mark.parametrize("dtype, scale", [
    (torch.float32, 1e-20),
    (torch.float64, 1e-150),
])
def test_similarity_conversion_avoids_determinant_underflow(convert, dtype, scale):
    matrix = torch.eye(3, dtype=dtype) * scale
    assert torch.det(matrix) == 0
    # The public tolerance is explicitly lowered to admit this nonzero scale.
    result = convert(matrix, atol=0)
    torch.testing.assert_close(result.scale(), torch.tensor([scale], dtype=dtype),
                               rtol=1e-5, atol=0)
    torch.testing.assert_close(result.rotation().matrix(), torch.eye(3, dtype=dtype))


@pytest.mark.parametrize("convert", [pp.mat2Sim3, pp.mat2RxSO3])
def test_similarity_conversion_large_scale_gradient(convert):
    log_scale = torch.tensor([300., 350.], dtype=torch.float64, requires_grad=True)
    matrix = log_scale.exp()[:, None, None] * torch.eye(3, dtype=log_scale.dtype)
    result = convert(matrix)
    gradient, = torch.autograd.grad(result.scale().log().sum(), log_scale)
    torch.testing.assert_close(gradient, torch.ones_like(log_scale))


@pytest.mark.parametrize("convert", [pp.mat2Sim3, pp.mat2RxSO3])
def test_similarity_conversion_retains_scale_tolerance(convert):
    with pytest.raises(ValueError, match="not full rank"):
        convert(torch.eye(3, dtype=torch.float64) * 1e-6)
    with pytest.raises(ValueError, match="not full rank"):
        convert(torch.zeros(3, 3, dtype=torch.float64))
    with pytest.raises(ValueError):
        convert(torch.diag(torch.tensor([-1., 1., 1.], dtype=torch.float64)))
