import pytest
import torch

import pypose as pp


@pytest.mark.parametrize(
    ("constructor", "values"),
    [
        (pp.SO3, [0.2, -0.3, 0.1, 0.4]),
        (pp.SE3, [0.2, -0.3, 0.1, 0.2, -0.3, 0.1, 0.4]),
        (pp.RxSO3, [0.2, -0.3, 0.1, 0.4, 1.2]),
        (pp.Sim3, [0.2, -0.3, 0.1, 0.2, -0.3, 0.1, 0.4, 1.2]),
    ],
)
def test_quat2unit_is_differentiable_for_leaf_lie_groups(constructor, values):
    source = torch.tensor(values, dtype=torch.float64, requires_grad=True)
    source_before = source.detach().clone()
    group = constructor(source)

    normalized = pp.quat2unit(group)
    loss = normalized.tensor().square().sum()
    loss.backward()

    torch.testing.assert_close(source.detach(), source_before)
    assert source.grad is not None
    assert torch.isfinite(source.grad).all()
    torch.testing.assert_close(
        normalized.rotation().tensor().norm(p=2),
        torch.ones((), dtype=source.dtype),
        atol=1e-12,
        rtol=1e-12,
    )


@pytest.mark.parametrize(
    ("constructor", "values", "quat_start"),
    [
        (pp.SO3, [[0.2, -0.3, 0.1, 0.4], [0.1, 0.2, -0.2, 0.3]], 0),
        (
            pp.SE3,
            [
                [0.2, -0.3, 0.1, 0.2, -0.3, 0.1, 0.4],
                [0.1, 0.2, -0.2, 0.1, 0.2, -0.1, 0.3],
            ],
            3,
        ),
        (
            pp.RxSO3,
            [[0.2, -0.3, 0.1, 0.4, 1.2], [0.1, 0.2, -0.2, 0.3, 0.8]],
            0,
        ),
        (
            pp.Sim3,
            [
                [0.2, -0.3, 0.1, 0.2, -0.3, 0.1, 0.4, 1.2],
                [0.1, 0.2, -0.2, 0.1, 0.2, -0.1, 0.3, 0.8],
            ],
            3,
        ),
    ],
)
def test_quat2unit_preserves_batched_non_quaternion_components(
    constructor, values, quat_start
):
    source = torch.tensor(values, dtype=torch.float64)
    group = constructor(source)
    normalized = pp.quat2unit(group).tensor()

    quat_end = quat_start + 4
    if quat_start:
        torch.testing.assert_close(
            normalized[..., :quat_start], source[..., :quat_start]
        )
    if quat_end < source.shape[-1]:
        torch.testing.assert_close(normalized[..., quat_end:], source[..., quat_end:])
    torch.testing.assert_close(
        normalized[..., quat_start:quat_end].norm(p=2, dim=-1),
        torch.ones(source.shape[:-1], dtype=source.dtype),
        atol=1e-12,
        rtol=1e-12,
    )
