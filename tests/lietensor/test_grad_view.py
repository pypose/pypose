"""Permanent Design-B regression tests: embedded storage + manifold grad view.

These tests pin down the PR #407 / Design B contract:

  * A Lie-group ``pp.Parameter`` keeps its ORIGINAL FULL EMBEDDED storage
    (SE3 = 7-wide: translation (3) + quaternion (4)). ``shape``, ``numel``,
    ``tensor()`` and ``state_dict`` are all embedded-width.
  * The public ``.grad`` transparently exposes the manifold (tangent)
    dimension (SE3 = 6) as a ZERO-COPY VIEW over the full embedded autograd
    gradient: it shares storage/data_ptr with the embedded gradient, and the
    single trailing redundant coordinate of the embedded gradient stays ~0.
  * The ``.grad`` setter accepts ``None`` (clear) and a manifold-width tensor
    (``torch.zeros_like(p.grad)``), zero-padding the redundant coordinate so
    the stored gradient stays embedded-width.
  * All differentiation APIs (``pp.func.jacobian``, ``pp.func.jacrev``,
    ``modjac``, ``modjacrev``, and the ``modjac`` flattened form) expose the
    manifold-width tangent axis (SE3 = 6).
  * PyPose optimizers (GaussNewton / LevenbergMarquardt) operate on tangent
    steps and Lie retraction: the parameter stays embedded-width with a unit
    quaternion and converges; a frozen parameter is left untouched.
  * Native ``torch.optim`` (Adam/SGD) is a documented OPEN/unsupported path:
    a 6-wide gradient cannot be applied as a 7-wide Euclidean update, so the
    step raises.

This replaces the earlier Design-A oriented tests (tangent-leaf 6-wide
parameter, working native Adam) that no longer reflect the implementation.
"""

import pytest
import torch
import pypose as pp
from torch import nn

SEED = 20260816
DEVICE = torch.device("cpu")
DTYPE = torch.float64

# (factory, group_type, embedding, manifold)
GROUP_CASES = [
    pytest.param(pp.randn_SO3, pp.SO3_type, 4, 3, id="SO3"),
    pytest.param(pp.randn_SE3, pp.SE3_type, 7, 6, id="SE3"),
    pytest.param(pp.randn_Sim3, pp.Sim3_type, 8, 7, id="Sim3"),
    pytest.param(pp.randn_RxSO3, pp.RxSO3_type, 5, 4, id="RxSO3"),
]


def _quaternion_is_unit(p, start, end):
    """The embedded [start:end] coordinates hold a unit quaternion."""
    q = p.tensor()[..., start:end]
    return bool(((q.norm(dim=-1) - 1.0).abs() < 1e-6).all())


# ---------------------------------------------------------------------------
# 1. Storage contract: embedded-width for shape/numel/tensor/state_dict
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("factory, group_type, embedding, manifold", GROUP_CASES)
def test_group_parameter_storage_is_embedded(factory, group_type, embedding,
                                             manifold):
    """A Lie-group Parameter keeps its ORIGINAL FULL EMBEDDED storage."""
    pose = pp.Parameter(factory(2, dtype=DTYPE, device=DEVICE))
    assert pose.ltype is group_type
    assert pose.shape == torch.Size([2, embedding]), str(pose.shape)
    assert pose.numel() == 2 * embedding, str(pose.numel())
    assert pose.tensor().shape == torch.Size([2, embedding])

    class W(nn.Module):
        def __init__(self):
            super().__init__()
            self.p = pose

    sd = W().state_dict()
    assert list(sd.values())[0].shape == torch.Size([2, embedding])


@pytest.mark.parametrize("factory, group_type, embedding, manifold", GROUP_CASES)
def test_algebra_tensor_is_manifold_width(factory, group_type, embedding,
                                          manifold):
    """The Lie-algebra (on-manifold) form is manifold-width, unchanged."""
    alg = factory(2, dtype=DTYPE, device=DEVICE).Log()
    assert alg.ltype.on_manifold
    assert alg.shape[-1] == manifold, str(alg.shape)
    assert alg.shape[-1] == int(alg.ltype.manifold[0])


# ---------------------------------------------------------------------------
# 2. grad contract: manifold-width zero-copy view over the embedded gradient
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("factory, group_type, embedding, manifold", GROUP_CASES)
def test_group_grad_is_manifold_width_zero_copy_view(factory, group_type,
                                                     embedding, manifold):
    """Public ``.grad`` is a manifold-width zero-copy view; the raw autograd
    gradient stays embedded-width with a ~0 redundant trailing coordinate."""
    torch.manual_seed(SEED)
    pts = torch.randn(2, 3, dtype=DTYPE, device=DEVICE)
    pose = pp.Parameter(factory(2, dtype=DTYPE, device=DEVICE))
    assert pose.grad is None, "grad must be None before backward"

    (pose @ pts).sum().backward()

    g = pose.grad
    raw = pose._raw_autograd_grad()
    assert g.shape == torch.Size([2, manifold]), str(g.shape)
    assert raw.shape == torch.Size([2, embedding]), str(raw.shape)
    # Zero-copy: the view aliases the embedded autograd gradient's storage.
    assert g.data_ptr() == raw.data_ptr()
    assert g.untyped_storage().data_ptr() == raw.untyped_storage().data_ptr()
    # The view is exactly the leading manifold slice of the embedded grad.
    assert torch.equal(g, raw[..., :manifold])
    # The single trailing redundant coordinate of the embedded gradient ~0.
    pad = raw[..., manifold:]
    assert pad.abs().max() < 1e-6, str(pad.flatten().tolist())


def test_grad_view_accumulation_and_rebuild():
    """Two backward() calls accumulate in the embedded slot; the view reflects
    the accumulation; ``p.grad = None`` clears and a fresh backward rebuilds a
    7-wide raw gradient with a 6-wide view."""
    torch.manual_seed(SEED)
    pts = torch.randn(1, 3, dtype=DTYPE, device=DEVICE)
    p = pp.Parameter(pp.randn_SE3(1, dtype=DTYPE, device=DEVICE))

    (p @ pts).sum().backward()
    g1 = p.grad.clone()
    (p @ pts).sum().backward()  # accumulate
    g2 = p.grad
    raw = p._raw_autograd_grad()
    assert torch.allclose(g2, 2.0 * g1, rtol=0, atol=1e-12)
    assert raw.shape == torch.Size([1, 7])
    assert (raw[..., 6:].abs() < 1e-9).all()

    p.grad = None
    assert p.grad is None and p._raw_autograd_grad() is None

    (p @ pts).sum().backward()
    assert p.grad.shape == torch.Size([1, 6])
    assert p._raw_autograd_grad().shape == torch.Size([1, 7])


@pytest.mark.parametrize("factory, group_type, embedding, manifold", GROUP_CASES)
def test_grad_print_and_repr_show_manifold_width(factory, group_type,
                                                  embedding, manifold):
    """A naive caller who does ``print(a.grad)`` (or ``str``/``repr``) with no
    knowledge of the embedded storage sees the manifold-width tensor, not the
    embedded one -- Chen's literal concern: printing must not leak the
    internal embedded layout."""
    torch.manual_seed(SEED)
    pts = torch.randn(2, 3, dtype=DTYPE, device=DEVICE)
    pose = pp.Parameter(factory(2, dtype=DTYPE, device=DEVICE))
    (pose @ pts).sum().backward()

    g = pose.grad
    assert g.shape == torch.Size([2, manifold]), str(g.shape)
    # print()/str()/repr() must not crash and must not leak the internal
    # LieTensor/embedded-type wrapper -- .grad is a plain Tensor view, so its
    # repr should look like an ordinary tensor, not "SE3Type LieTensor: ...".
    printed = str(g)
    reprd = repr(g)
    assert "LieTensor" not in printed, printed
    assert "LieTensor" not in reprd, reprd
    # The printed value has exactly `manifold` numbers per row, not `embedding`.
    first_row = printed.split("\n")[0]
    assert first_row.count(".") == manifold or first_row.count(",") in (
        manifold - 1, manifold), printed


@pytest.mark.parametrize("factory, group_type, embedding, manifold", GROUP_CASES)
def test_grad_view_inplace_mutation_propagates_to_raw_storage(
        factory, group_type, embedding, manifold):
    """Answering Chen's core question directly: since ``.grad`` is a
    genuine zero-copy VIEW (not a copy, and not an independently
    reassignable shape/dimension property), a user cannot "cheat" by
    mutating the view without it being reflected in the real, full-width
    autograd storage -- there is no separate faked-shape state to diverge
    from the actual gradient. An in-place edit through the manifold-width
    view is visible immediately in the embedded-width raw storage."""
    if manifold == embedding:
        pytest.skip("this group has no redundant embedded coordinate")
    torch.manual_seed(SEED)
    pts = torch.randn(2, 3, dtype=DTYPE, device=DEVICE)
    pose = pp.Parameter(factory(2, dtype=DTYPE, device=DEVICE))
    (pose @ pts).sum().backward()

    with torch.no_grad():
        pose.grad[0, 0] = 12345.0
    raw = pose._raw_autograd_grad()
    assert raw[0, 0].item() == 12345.0, \
        "mutating the view must mutate the real embedded storage in place"
    assert pose.grad[0, 0].item() == 12345.0


# ---------------------------------------------------------------------------
# 3. grad setter: None / manifold-width (zeros_like) / embedded-width
# ---------------------------------------------------------------------------

def test_grad_setter_none_and_zeros_like_zero_pad():
    """``p.grad = None`` clears; ``p.grad = torch.zeros_like(p.grad)`` (a
    manifold-width assignment) writes zeros into the leading coordinates and
    zero-pads the redundant trailing coordinate, keeping the stored gradient
    embedded-width."""
    torch.manual_seed(SEED)
    pts = torch.randn(1, 3, dtype=DTYPE, device=DEVICE)
    p = pp.Parameter(pp.randn_SE3(1, dtype=DTYPE, device=DEVICE))
    (p @ pts).sum().backward()

    p.grad = None
    assert p.grad is None and p._raw_autograd_grad() is None

    (p @ pts).sum().backward()
    p.grad = torch.zeros_like(p.grad)  # 6-wide assignment
    raw = p._raw_autograd_grad()
    assert raw.shape == torch.Size([1, 7]), str(raw.shape)
    assert (p.grad == 0).all()
    assert (raw[..., 6:].abs() < 1e-12).all()


def test_grad_setter_embedded_width_and_tangent_grad_alias():
    """An embedded-width assignment is stored unchanged; the ``tangent_grad``
    alias equals the public ``grad`` view."""
    torch.manual_seed(SEED)
    pts = torch.randn(1, 3, dtype=DTYPE, device=DEVICE)
    p = pp.Parameter(pp.randn_SE3(1, dtype=DTYPE, device=DEVICE))
    (p @ pts).sum().backward()

    v = torch.randn(1, 7, dtype=DTYPE, device=DEVICE)
    p.grad = v
    assert torch.equal(p._raw_autograd_grad(), v)
    assert torch.equal(p.tangent_grad, p.grad)


def test_algebra_and_ordinary_tensor_grad_passthrough():
    """Lie-algebra (on-manifold) and ordinary Tensor parameters keep the
    unmodified autograd grad shape (no slicing is applied to them)."""
    torch.manual_seed(SEED)
    tn = pp.Parameter(torch.randn(1, 3, dtype=DTYPE, device=DEVICE))
    tn.sum().backward()
    assert tn.grad.shape == torch.Size([1, 3])

    th = pp.Parameter(pp.randn_se3(1, dtype=DTYPE, device=DEVICE))
    th.sum().backward()
    assert th.grad.shape == torch.Size([1, 6])

    nong = pp.Parameter(pp.randn_SE3(1, dtype=DTYPE, device=DEVICE))
    assert nong.grad is None, "grad is None before backward"


# ---------------------------------------------------------------------------
# 4. Differentiation APIs all expose the manifold-width tangent axis
# ---------------------------------------------------------------------------

def test_differentiation_apis_expose_tangent_axis():
    """jacobian / jacrev / modjac / modjacrev all expose the 6-wide tangent
    axis for an SE3 Parameter."""
    torch.manual_seed(SEED)

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.p = pp.Parameter(pp.randn_SE3(2, dtype=DTYPE, device=DEVICE))

        def forward(self, x):
            return self.p @ x

    model = M()
    x = torch.randn(2, 3, dtype=DTYPE, device=DEVICE)

    J = pp.func.jacobian(lambda q, a: q @ a, (model.p, x), strict=True)
    Jp = J[0] if isinstance(J, tuple) else J
    assert Jp.shape[-1] == 6, str(Jp.shape)

    J = pp.func.jacrev(lambda q, a: q @ a, argnums=0)(model.p, x)
    assert J.shape[-1] == 6, str(J.shape)

    J = pp.optim.functional.modjac(model, x)
    J = J[0] if isinstance(J, tuple) else J
    assert J.shape[-1] == 6, str(J.shape)

    J = pp.optim.functional.modjacrev(model, x)
    assert J["p"].shape[-1] == 6, str(J["p"].shape)


def test_modjac_flatten_is_manifold_width_columns():
    """``modjac(..., flatten=True)`` yields N*manifold columns (SE3: N*6)."""
    torch.manual_seed(SEED)

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.p = pp.Parameter(pp.randn_SE3(3, dtype=DTYPE, device=DEVICE))

        def forward(self, x):
            return self.p @ x

    model = M()
    x = torch.randn(3, 3, dtype=DTYPE, device=DEVICE)
    J = pp.optim.functional.modjac(model, x, flatten=True)
    assert J.shape == torch.Size([9, 18]), str(J.shape)


def test_modjac_multi_output_keeps_all_outputs_tangent_axis():
    """A model returning a tuple of outputs yields a per-output Jacobian
    structure (the jacobian() layout for multi-output models), and EVERY
    output block keeps the manifold-width tangent axis. Regression test:
    the tangent-axis slice must not truncate the per-output structure (a
    naive zip of the top level against the params silently dropped all but
    the first output and broke LM multi-output steps with an IndexError)."""
    torch.manual_seed(SEED)

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.p = pp.Parameter(pp.randn_SE3(2, 2, dtype=DTYPE, device=DEVICE))

        def forward(self, x):
            # error1: per-pose Lie-algebra error, (2,2,2,2,6) -> 96 scalar rows
            # error2: scalar per pose,           (2,2,1)     ->  4 scalar rows
            error1 = (self.p @ x).Log().tensor()
            error2 = self.p.Log().tensor().sum(-1, keepdim=True)
            return error1, error2

    model = M()
    x = pp.randn_SE3(2, 2, 2, 2, dtype=DTYPE, device=DEVICE)

    J = pp.optim.functional.modjac(model, x, flatten=False)
    assert isinstance(J, tuple) and len(J) == 2, "both outputs must be kept"
    for Jr in J:
        Jr = Jr[0] if isinstance(Jr, tuple) else Jr
        assert Jr.shape[-1] == 6, str(Jr.shape)

    Jf = pp.optim.functional.modjac(model, x, flatten=True)
    # rows = error1 rows + error2 rows; cols = 4 poses x 6 tangent coordinates
    n_rows = 2 * 2 * 2 * 2 * 6 + 2 * 2 * 1
    assert Jf.shape == torch.Size([n_rows, 2 * 2 * 6]), str(Jf.shape)


def test_modjacrev_has_aux_tangent_axis():
    """``modjacrev(..., has_aux=True)`` returns the aux alongside a 6-wide
    parameter Jacobian."""
    torch.manual_seed(SEED)

    class AuxModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.p = pp.Parameter(pp.randn_SE3(2, dtype=DTYPE, device=DEVICE))

        def forward(self, x):
            out = self.p @ x
            return out, out[..., 0].sum()

    model = AuxModel()
    x = torch.randn(2, 3, dtype=DTYPE, device=DEVICE)
    J, aux = pp.optim.functional.modjacrev(model, x, has_aux=True)
    assert aux is not None
    assert J["p"].shape[-1] == 6, str(J["p"].shape)


# ---------------------------------------------------------------------------
# 5. PyPose optimizers: embedded storage + unit quaternion + convergence
# ---------------------------------------------------------------------------

def _se3_reprojection_model():
    """Build an SE3 parameter (4 poses, each applied to its 3 points) and a
    ground-truth observation, returning (model, x0, obs)."""
    torch.manual_seed(SEED)
    x0 = torch.randn(4, 3, 3, dtype=DTYPE, device=DEVICE)
    pstar = pp.randn_SE3(4, dtype=DTYPE, device=DEVICE)
    obs = torch.stack([pstar[i:i + 1] @ x0[i] for i in range(4)])

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.p = pp.Parameter(
                pp.randn_SE3(4, dtype=DTYPE, device=DEVICE))

        def forward(self, x):
            # Per-pose application: pypose's broadcast_inputs right-aligns
            # batch dims, so (4,7) @ (4,3,3) cannot pair pose i with points i.
            return torch.stack(
                [self.p[i:i + 1] @ x[i] for i in range(x.shape[0])])

    return M(), x0, obs


@pytest.mark.parametrize("make_opt",
                         [pp.optim.GaussNewton, pp.optim.LM],
                         ids=["GaussNewton", "LevenbergMarquardt"])
def test_pypose_optimizer_keeps_embedded_and_converges(make_opt):
    """GaussNewton / LevenbergMarquardt keep the SE3 Parameter 7-wide with a
    unit quaternion, decrease the loss, and recover the ground truth."""
    model, x0, obs = _se3_reprojection_model()
    opt = make_opt(model)
    losses = [float(opt.step(x0, obs)) for _ in range(10)]

    assert model.p.shape == torch.Size([4, 7]), str(model.p.shape)
    assert _quaternion_is_unit(model.p, 3, 7)
    assert losses[-1] < losses[0], "loss must decrease"

    resid = torch.stack(
        [model.p[i:i + 1] @ x0[i] for i in range(4)]) - obs
    assert resid.norm() < 1e-8, "residual=%.3e" % resid.norm().item()


def test_pypose_optimizer_frozen_parameter_untouched():
    """A frozen (requires_grad=False) SE3 Parameter is left untouched while an
    active SE3 Parameter is optimized to completion by LevenbergMarquardt."""
    torch.manual_seed(SEED)
    x = torch.randn(1, 3, dtype=DTYPE, device=DEVICE)
    pf = pp.randn_SE3(1, dtype=DTYPE, device=DEVICE)
    pa = pp.randn_SE3(1, dtype=DTYPE, device=DEVICE)
    obs = pf @ (pa @ x)

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.frozen = pp.Parameter(
                pp.randn_SE3(1, dtype=DTYPE, device=DEVICE))
            self.frozen.requires_grad_(False)
            self.active = pp.Parameter(
                pp.randn_SE3(1, dtype=DTYPE, device=DEVICE))

        def forward(self, x):
            return self.frozen @ (self.active @ x)

    m = M()
    with torch.no_grad():
        m.frozen.tensor().copy_(pf.tensor())
    frozen_before = m.frozen.tensor().clone()
    resid0 = (m.frozen @ (m.active @ x)) - obs

    opt = pp.optim.LM(m)
    losses = [float(opt.step(x, obs)) for _ in range(20)]

    assert torch.equal(m.frozen.tensor(), frozen_before), \
        "frozen parameter must be untouched"
    assert m.active.shape == torch.Size([1, 7])
    assert _quaternion_is_unit(m.active, 3, 7)
    assert losses[-1] < losses[0], "loss must decrease"
    resid = (m.frozen @ (m.active @ x)) - obs
    assert resid.norm() < resid0.norm() * 1e-2, \
        "residual must be reduced by >= 100x"


# ---------------------------------------------------------------------------
# 6. Native torch.optim is a documented OPEN/unsupported limitation
# ---------------------------------------------------------------------------

def test_native_torch_optim_adam_is_open_unsupported():
    """torch.optim.Adam on an SE3 group Parameter raises a size mismatch: the
    6-wide gradient cannot be applied as a 7-wide Euclidean update (Adam uses
    the non-overridden in-place ``addcdiv_``). This is the documented OPEN
    limitation -- use a PyPose optimizer for LieTensor parameters.
    """
    torch.manual_seed(SEED)
    p = pp.Parameter(pp.randn_SE3(1, dtype=DTYPE, device=DEVICE))
    pts = torch.randn(1, 3, dtype=DTYPE, device=DEVICE)
    (p @ pts).sum().backward()
    assert p.grad.shape == torch.Size([1, 6])
    opt = torch.optim.Adam([p], lr=1e-3)
    with pytest.raises(RuntimeError, match=r"size of tensor"):
        opt.step()
    # The parameter storage is untouched by the failed step.
    assert p.shape == torch.Size([1, 7])


def test_native_torch_optim_sgd_hits_preexisting_add_override():
    """SGD's ``param.add_(grad, alpha=-lr)`` is intercepted by pypose's
    PRE-EXISTING ``LieTensor.add_`` override (``Exp(other) * input``), which
    performs a manifold retraction rather than the raw 7-vs-6 Euclidean add.
    So SGD does NOT raise here; this documents the pre-existing interaction
    rather than a Design-B contract. The native path is still unsupported in
    general (see the Adam test) -- use a PyPose optimizer for LieTensor params.
    """
    torch.manual_seed(SEED)
    p = pp.Parameter(pp.randn_SE3(1, dtype=DTYPE, device=DEVICE))
    pts = torch.randn(1, 3, dtype=DTYPE, device=DEVICE)
    (p @ pts).sum().backward()
    assert p.grad.shape == torch.Size([1, 6])
    before = p.tensor().clone()
    torch.optim.SGD([p], lr=1e-3).step()
    # The pre-existing add_ override performs a retraction: the parameter
    # moves but stays a valid 7-wide embedded group (unit quaternion).
    assert not torch.equal(p.tensor(), before), \
        "SGD via the pre-existing add_ override performs a retraction step"
    assert p.shape == torch.Size([1, 7])
    assert _quaternion_is_unit(p, 3, 7)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
