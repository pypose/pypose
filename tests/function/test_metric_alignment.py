import pytest
import torch
import pypose as pp
import pypose.metric.ape_rpe as metric


@pytest.mark.parametrize('name', ['ape', 'rpe'])
@pytest.mark.parametrize('nposes', [2, 3, 6, -1])
@pytest.mark.parametrize('mode', ['align', 'scale'])
def test_alignment_uses_pose_subset(monkeypatch, name, nposes, mode):
    stamps = torch.arange(6, dtype=torch.float64)
    poses = pp.identity_SE3(6, dtype=torch.float64)
    positions = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0],
                              [0, 0, 1], [2, 1, 0], [3, 0, 1]], dtype=torch.float64)
    poses.tensor()[:, :3] = positions
    expected = positions if nposes == -1 else positions[:nposes]
    original = metric.svdstf
    calls = []

    def checked_alignment(source, target, scale):
        calls.append((source, target))
        torch.testing.assert_close(source, expected)
        torch.testing.assert_close(target, expected)
        return original(source, target, scale)

    monkeypatch.setattr(metric, 'svdstf', checked_alignment)
    getattr(pp.metric, name)(stamps.clone(), poses, stamps.clone(), poses,
                            nposes=nposes, **{mode: True})
    assert len(calls) == 1


def test_ape_alignment_ignores_poses_after_subset():
    stamps = torch.arange(6, dtype=torch.float64)
    reference = pp.identity_SE3(6, dtype=torch.float64)
    reference.tensor()[:, :3] = torch.tensor(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [2, 1, 0], [3, 0, 1]],
        dtype=torch.float64)
    estimated = reference.clone()
    estimated.tensor()[3:, 0] += 10
    error = pp.metric.ape(stamps.clone(), reference, stamps.clone(), estimated,
                          align=True, nposes=3, otype='Mean')
    torch.testing.assert_close(error, torch.tensor(5., dtype=torch.float64))
