import pytest
import torch
import pypose as pp
from torch import nn
import pypose.optim.solver as ppos

from pypose.optim.optimizer import BAE_AVAILABLE


def _get_tracking_tensor():
    try:
        from bae.autograd.function import TrackingTensor
    except Exception:
        return None
    return TrackingTensor


class _SparseIdentityModel(nn.Module):
    def __init__(self, x0, TrackingTensor):
        super().__init__()
        self.x = nn.Parameter(TrackingTensor(x0))

    def forward(self):
        return self.x


class TestSparseLM:
    def test_sparse_lm_runs_and_converges(self):
        if not BAE_AVAILABLE:
            pytest.skip("sparse LM backend unavailable (pypose.optim.optimizer.BAE_AVAILABLE is False)")

        TrackingTensor = _get_tracking_tensor()
        if TrackingTensor is None:
            pytest.skip("bae is required for sparse LM (TrackingTensor unavailable)")
        if not torch.cuda.is_available():
            pytest.skip("sparse LM backend requires CUDA")

        torch.manual_seed(0)
        device = torch.device("cuda")
        dtype = torch.float64

        x_true = torch.randn(8, 1, device=device, dtype=dtype)
        x0 = x_true + 0.1 * torch.randn_like(x_true)

        model = _SparseIdentityModel(x0, TrackingTensor).to(device)
        target = x_true

        strategy = pp.optim.strategy.Constant(damping=1e-6)
        try:
            optimizer = pp.optim.LM(model, solver=ppos.CG(), strategy=strategy, sparse=True)
        except Exception as e:
            msg = str(e).lower()
            if "cuda" in msg or "cusparse" in msg:
                pytest.skip(f"sparse LM backend unavailable on this machine: {e}")
            raise

        with torch.no_grad():
            loss0 = optimizer.model.loss(input=(), target=target).item()

        loss = loss0
        try:
            for _ in range(6):
                loss = optimizer.step(input=(), target=target).item()
        except Exception as e:
            msg = str(e).lower()
            if "cuda" in msg or "cusparse" in msg:
                pytest.skip(f"sparse LM backend unavailable on this machine: {e}")
            raise

        assert loss < loss0
        torch.testing.assert_close(model.x.tensor(), x_true, rtol=1e-4, atol=1e-4)
