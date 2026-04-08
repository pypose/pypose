import pytest
import torch
import pypose as pp
import pypose.autograd.function as ppaf
from torch import nn
import pypose.optim.solver as ppos
from pypose.autograd.function import Track, parallel_for_sparse_jacobian


@parallel_for_sparse_jacobian
def edge_error(node1, node2, relpose):
    return (relpose.Inv() @ node1.Inv() @ node2).Log().tensor()


class _SparseIdentityModel(nn.Module):
    def __init__(self, x0):
        super().__init__()
        self.x = nn.Parameter(Track(x0))

    def forward(self):
        return self.x


class _SparseChainPGO(nn.Module):
    def __init__(self, root, nodes):
        super().__init__()
        self.register_buffer("root", root)
        self.nodes = nn.Parameter(Track(nodes))

    def forward(self, edges, relposes):
        nodes = torch.cat((self.root, self.nodes), dim=0)
        return edge_error(
            nodes[edges[:, 0]],
            nodes[edges[:, 1]],
            relposes,
        )


class TestSparseLM:
    def test_track_alias(self):
        assert Track is ppaf.TrackingTensor

    def test_sparse_lm_runs_and_converges(self):
        if not torch.cuda.is_available():
            pytest.skip("sparse LM backend requires CUDA")

        torch.manual_seed(0)
        device = torch.device("cuda")
        dtype = torch.float64

        x_true = torch.randn(8, 1, device=device, dtype=dtype)
        x0 = x_true + 0.1 * torch.randn_like(x_true)

        try:
            model = _SparseIdentityModel(x0).to(device)
            optimizer = pp.optim.LM(
                model,
                solver=ppos.PCG(),
                strategy=pp.optim.strategy.Constant(damping=1e-6),
                sparse=True,
            )
            target = x_true
        except ImportError as e:
            pytest.skip(f"sparse LM backend unavailable on this machine: {e}")
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

    def test_sparse_lm_chain_pgo_runs_and_converges(self):
        if not torch.cuda.is_available():
            pytest.skip("sparse LM backend requires CUDA")

        torch.manual_seed(0)
        device = torch.device("cuda")
        dtype = torch.float64

        gt_nodes = pp.SE3(
            torch.tensor(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                ],
                device=device,
                dtype=dtype,
            )
        )
        edges = torch.tensor([[0, 1], [1, 2]], device=device)
        relposes = gt_nodes[edges[:, 0]].Inv() @ gt_nodes[edges[:, 1]]
        init = gt_nodes[1:] * pp.randn_SE3(2, sigma=0.1, device=device, dtype=dtype)

        try:
            model = _SparseChainPGO(gt_nodes[:1], init).to(device)
            optimizer = pp.optim.LM(
                model,
                solver=ppos.PCG(),
                strategy=pp.optim.strategy.Constant(damping=1e-4),
                sparse=True,
            )
        except ImportError as e:
            pytest.skip(f"sparse LM backend unavailable on this machine: {e}")
        except Exception as e:
            msg = str(e).lower()
            if "cuda" in msg or "cusparse" in msg:
                pytest.skip(f"sparse LM backend unavailable on this machine: {e}")
            raise

        with torch.no_grad():
            loss0 = optimizer.model.loss(input=(edges, relposes), target=None).item()

        loss = loss0
        try:
            for _ in range(5):
                loss = optimizer.step(input=(edges, relposes)).item()
                if loss < 1e-5:
                    break
        except Exception as e:
            msg = str(e).lower()
            if "cuda" in msg or "cusparse" in msg:
                pytest.skip(f"sparse LM backend unavailable on this machine: {e}")
            raise

        assert loss < loss0
        assert loss < 1e-5
        torch.testing.assert_close(
            pp.SE3(model.nodes).translation(),
            gt_nodes[1:].translation(),
            rtol=1e-4,
            atol=2e-4,
        )
