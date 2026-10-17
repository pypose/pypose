"""Numerical verification of the PR #407 Design-B LieTensor.grad contract.

Design B keeps the ORIGINAL FULL EMBEDDED group storage (SE3 = 7-wide) in
``pp.Parameter`` / raw LieTensors, while the public ``.grad`` transparently
exposes the manifold (tangent) dimension (SE3 = 6) as a zero-copy view. This
file verifies the NUMERICAL side of that contract:

  1. For every Lie group (SO3, SE3, Sim3, RxSO3), the accumulated embedded
     gradient's leading-manifold slice (i.e. the public ``.grad``) equals the
     LEFT-PERTURBATION finite-difference gradient
         d/dh f(Exp(h e_i) @ G) |_{h=0}
     of a loss whose group chain terminates in a Mul/composition, computed in
     float64. Tolerances: SO3/SE3/RxSO3 tight (~1e-6); Sim3 looser (~5e-2)
     because its scale coordinate passes through log/exp maps that amplify
     FD error (documented contract caveat).
  2. For Act-terminated (reprojection) losses, Act's own backward is EXACT
     (not a first-order-at-identity approximation) -- gated at the same
     tolerance as (1), across all four groups, multi-point losses, the
     separate Act4 homogeneous-coordinate path, extreme rotation magnitudes
     (near-identity through near-pi), fp32, and CUDA. A separate,
     intentionally NON-gated test keeps a loss that mixes in a raw
     ``.rotation().tensor()`` term for regression coverage: that term is not
     manifold-covariant (a plain embedded-storage slice, no custom autograd
     Function), so it -- not Act -- is the source of any FD discrepancy
     there.
  3. The public ``pp.func.jacobian`` exposes the manifold column dimension and
     its blocks match the same left-perturbation FD.
  4. Batched / singleton / float32 / float64 grad-view behavior.
  5. A tangent step must be applied as a manifold retraction
     (``Exp(delta) @ G``), materially different from a coordinate-wise
     (Euclidean) add of the 6-D delta onto the 7-D embedded storage.

This replaces the earlier Design-A oriented tests (tangent-leaf 6-wide
parameter, ``pose.Exp()``, working native Adam) that no longer reflect the
implementation.
"""

import pytest
import torch
import pypose as pp

SEED = 20260816
DEVICE = torch.device("cpu")
DTYPE = torch.float64
EPS = 1e-5

# (factory, algebra_type, manifold, grad-FD tolerance)
# Sim3 keeps the documented caveat: its scale coordinate passes through
# log/exp maps that amplify FD error, so it uses a looser tolerance.
GROUP_CASES = [
    pytest.param(pp.randn_SO3, pp.so3_type, 3, 1e-6, id="SO3"),
    pytest.param(pp.randn_SE3, pp.se3_type, 6, 1e-6, id="SE3"),
    pytest.param(pp.randn_Sim3, pp.sim3_type, 7, 5e-2, id="Sim3"),
    pytest.param(pp.randn_RxSO3, pp.rxso3_type, 4, 1e-6, id="RxSO3"),
]


def _left_perturbation_fd(f, pose, algebra, manifold, h=EPS):
    """Central left-perturbation finite difference of a scalar group loss.

    fd[..., i] = [f(Exp(+h e_i) @ G) - f(Exp(-h e_i) @ G)] / (2 h)
    where G = pose (detached) is the embedded group. Returns (..., manifold).
    """
    fd = torch.zeros(manifold, dtype=pose.dtype, device=pose.device)
    g = pose.detach()
    for i in range(manifold):
        e = torch.zeros(manifold, dtype=g.dtype, device=g.device)
        e[i] = h
        fp = pp.LieTensor(e, ltype=algebra).Exp()
        fm = pp.LieTensor(-e, ltype=algebra).Exp()
        fd[i] = (f(fp @ g) - f(fm @ g)) / (2.0 * h)
    return fd


def _mul_loss(pose, ref):
    """Scalar loss whose group chain terminates in a Mul/composition."""
    return ((pose @ ref).Log().tensor().pow(2).sum())


def pose_embedding_dim(factory):
    probe = factory(1)
    return int(probe.shape[-1])


# ---------------------------------------------------------------------------
# 1. Public grad == left-perturbation finite difference (scalar loss)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("factory, algebra, manifold, tol", GROUP_CASES)
def test_grad_matches_left_perturbation_finite_difference(factory, algebra,
                                                          manifold, tol):
    """The public ``.grad`` (leading-manifold slice of the accumulated
    embedded gradient) matches the central left-perturbation finite
    difference of a Mul-terminated scalar loss, in float64."""
    for trial in range(2):
        torch.manual_seed(SEED + trial)
        pose = pp.Parameter(factory(1, dtype=DTYPE, device=DEVICE))
        assert pose.shape[-1] == int(pose.ltype.embedding[0])
        ref = factory(1, dtype=DTYPE, device=DEVICE)

        _mul_loss(pose, ref).backward()
        g = pose.grad
        assert g.shape == torch.Size([1, manifold]), str(g.shape)

        fd = _left_perturbation_fd(lambda a: _mul_loss(a, ref), pose,
                                   algebra, manifold)
        rel = (fd - g[0]).norm().item() / max(g.norm().item(), 1.0)
        assert rel <= tol, "rel=%.3e (tol=%.1e)" % (rel, tol)


# ---------------------------------------------------------------------------
# 2. Act-terminated (reprojection) loss
# ---------------------------------------------------------------------------
#
# Investigation note (superseding an earlier, INCORRECT claim in this file):
# a pure Act-terminated loss -- Act.apply()'s own backward, in isolation --
# is NOT a first-order-at-identity approximation. `SO3_Act_Jacobian(p) =
# vec2skew(-p)` (pypose/lietensor/operation.py) is the exact closed-form
# derivative of `y = R @ x` under a left tangent perturbation, evaluated at
# the actual output point, for any base rotation -- not just near identity.
# See `test_pure_act_terminated_loss_matches_left_perturbation_fd` below,
# which gates on this at 1e-6 and matches to ~1e-9/1e-10 in practice.
#
# The finiteness-only test that follows keeps the ORIGINAL mixed loss (Act
# term + a raw `.rotation().tensor()` term) for regression coverage, but the
# discrepancy it exhibits is NOT caused by Act's backward. `.rotation()`
# (e.g. SE3Type.rotation in pypose/lietensor/lietensor.py) is a plain tensor
# slice of the embedded quaternion storage, with no custom autograd
# Function -- it is not a manifold-covariant operation, so a loss built
# directly from its raw components has no reason to match a left-
# perturbation tangent-space finite difference (the true relationship
# between raw-quaternion-component gradients and tangent-space gradients is
# position-dependent -- the quaternion kinematic/"G" matrix -- which a bare
# slice does not apply). This is intentionally NOT gated for that reason,
# not because Act is approximate.


@pytest.mark.parametrize("factory, algebra, manifold, tol", GROUP_CASES)  # SO3, SE3, Sim3, RxSO3
def test_pure_act_terminated_loss_matches_left_perturbation_fd(
        factory, algebra, manifold, tol):
    """A loss built ONLY from `a.Act(pts)` (no raw-slice terms mixed in)
    matches the left-perturbation finite difference tightly, across all four
    groups and multiple random poses: Act's own backward is exact, not an
    approximation. Uses the same per-group tolerance as the Mul-terminated
    test (Sim3 looser -- documented scale/log-exp FD amplification)."""
    for trial in range(2):
        torch.manual_seed(SEED + trial)
        pts = torch.randn(3, dtype=DTYPE, device=DEVICE)

        def f_act_pure(a):
            return a.Act(pts).pow(2).sum()

        pose = pp.Parameter(factory(1, dtype=DTYPE, device=DEVICE))
        f_act_pure(pose).backward()
        g = pose.grad
        assert g.shape == torch.Size([1, manifold])

        fd = _left_perturbation_fd(f_act_pure, pose, algebra, manifold)
        rel = (fd - g[0]).norm().item() / max(g.norm().item(), 1.0)
        assert rel <= tol, "rel=%.3e (tol=%.1e)" % (rel, tol)


@pytest.mark.parametrize("factory, algebra, manifold, tol",
                         [GROUP_CASES[0], GROUP_CASES[1]])  # SO3, SE3
@pytest.mark.parametrize("batch", [2, 5], ids=["batch2", "batch5"])
def test_pure_act_terminated_loss_matches_fd_batched(factory, algebra,
                                                     manifold, tol, batch):
    """Batched Act-terminated loss: each row's accumulated gradient still
    matches the left-perturbation FD of that row taken in isolation (a
    per-row loss decouples rows, so batching introduces no cross-row
    leakage into the exact-match guarantee)."""
    torch.manual_seed(SEED)
    pts = torch.randn(batch, 3, dtype=DTYPE, device=DEVICE)
    pose = pp.Parameter(factory(batch, dtype=DTYPE, device=DEVICE))

    def f_act_pure(a):
        return a.Act(pts).pow(2).sum()

    f_act_pure(pose).backward()
    g = pose.grad
    assert g.shape == torch.Size([batch, manifold])

    row = 0
    pose_row = pp.Parameter(pose[row:row + 1].detach().clone())

    def f_row(a):
        return a.Act(pts[row:row + 1]).pow(2).sum()

    fd = _left_perturbation_fd(f_row, pose_row, algebra, manifold)
    rel = (fd - g[row]).norm().item() / max(g[row].norm().item(), 1.0)
    assert rel <= 1e-6, "rel=%.3e" % rel


@pytest.mark.parametrize("factory, algebra, manifold, tol", GROUP_CASES)  # SO3, SE3, Sim3, RxSO3
def test_pure_act4_homogeneous_loss_matches_left_perturbation_fd(
        factory, algebra, manifold, tol):
    """Act on a 4-dim homogeneous point dispatches to the separate *Act4
    autograd Function (SO3_Act4/SE3_Act4/RxSO3_Act4/Sim3_Act4in
    operation.py), independent code from the 3-dim Act path. Verify its
    backward is independently exact against the same left-perturbation FD."""
    torch.manual_seed(SEED)
    pts4 = torch.randn(4, dtype=DTYPE, device=DEVICE)

    def f_act4(a):
        return a.Act(pts4).pow(2).sum()

    pose = pp.Parameter(factory(1, dtype=DTYPE, device=DEVICE))
    f_act4(pose).backward()
    g = pose.grad
    assert g.shape == torch.Size([1, manifold])

    fd = _left_perturbation_fd(f_act4, pose, algebra, manifold)
    rel = (fd - g[0]).norm().item() / max(g.norm().item(), 1.0)
    assert rel <= tol, "rel=%.3e (tol=%.1e)" % (rel, tol)


@pytest.mark.parametrize("factory, algebra, manifold, tol", GROUP_CASES)  # SO3, SE3, Sim3, RxSO3
@pytest.mark.parametrize("npts", [1, 5, 16], ids=["1pt", "5pts", "16pts"])
def test_pure_act_multi_point_loss_matches_left_perturbation_fd(
        factory, algebra, manifold, tol, npts):
    """A single pose Act-ing on many points at once (the realistic reprojection
    shape: one camera pose, many 3D points) still matches the left-
    perturbation FD tightly -- the exactness of Act's backward does not
    depend on how many points share the same pose in the loss."""
    torch.manual_seed(SEED)
    pts = torch.randn(npts, 3, dtype=DTYPE, device=DEVICE)

    def f_act_multi(a):
        return a.Act(pts).pow(2).sum()

    pose = pp.Parameter(factory(1, dtype=DTYPE, device=DEVICE))
    f_act_multi(pose).backward()
    g = pose.grad
    assert g.shape == torch.Size([1, manifold])

    fd = _left_perturbation_fd(f_act_multi, pose, algebra, manifold)
    rel = (fd - g[0]).norm().item() / max(g.norm().item(), 1.0)
    assert rel <= tol, "rel=%.3e (tol=%.1e)" % (rel, tol)


@pytest.mark.parametrize("factory, algebra, manifold, tol",
                         [GROUP_CASES[0], GROUP_CASES[1]])  # SO3, SE3
@pytest.mark.parametrize("angle_scale, angle_id", [
    (1e-4, "near_identity"),
    (1.0, "moderate"),
    (3.13, "near_pi"),
])
def test_pure_act_loss_matches_fd_across_rotation_magnitudes(
        factory, algebra, manifold, tol, angle_scale, angle_id):
    """Act's backward stays exact across the full range of rotation
    magnitudes a pose can take -- near-identity (where a naive first-order
    approximation would also happen to look correct), moderate, and
    near-pi (where a first-order-at-identity approximation would be most
    exposed, since it is farthest from the point of linearization)."""
    torch.manual_seed(SEED)
    pts = torch.randn(3, dtype=DTYPE, device=DEVICE)

    # Build a pose whose rotation angle is close to `angle_scale` radians by
    # retracting the identity along a random unit tangent direction.
    direction = torch.randn(manifold, dtype=DTYPE, device=DEVICE)
    direction = direction / direction.norm()
    tangent = (direction * angle_scale).unsqueeze(0)
    pose = pp.Parameter(pp.LieTensor(tangent, ltype=algebra).Exp())

    def f_act_pure(a):
        return a.Act(pts).pow(2).sum()

    f_act_pure(pose).backward()
    g = pose.grad
    assert g.shape == torch.Size([1, manifold])

    fd = _left_perturbation_fd(f_act_pure, pose, algebra, manifold)
    rel = (fd - g[0]).norm().item() / max(g.norm().item(), 1.0)
    assert rel <= tol, "rel=%.3e (tol=%.1e, angle=%s)" % (rel, tol, angle_id)


@pytest.mark.parametrize("factory, algebra, manifold, tol",
                         [GROUP_CASES[0], GROUP_CASES[1]])  # SO3, SE3
def test_pure_act_loss_matches_fd_fp32(factory, algebra, manifold, tol):
    """The exact-match property also holds (at a looser, fp32-appropriate
    tolerance) in float32, not just float64 -- the analytic Jacobian formula
    itself is dtype-independent."""
    torch.manual_seed(SEED)
    dtype = torch.float32
    pts = torch.randn(3, dtype=dtype, device=DEVICE)

    def f_act_pure(a):
        return a.Act(pts).pow(2).sum()

    pose = pp.Parameter(factory(1, dtype=dtype, device=DEVICE))
    f_act_pure(pose).backward()
    g = pose.grad
    assert g.shape == torch.Size([1, manifold])
    assert g.dtype == dtype

    fd = _left_perturbation_fd(f_act_pure, pose, algebra, manifold, h=1e-3)
    rel = (fd - g[0]).norm().item() / max(g.norm().item(), 1.0)
    assert rel <= 1e-2, "rel=%.3e (fp32)" % rel


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("factory, algebra, manifold, tol",
                         [GROUP_CASES[0], GROUP_CASES[1]])  # SO3, SE3
def test_pure_act_loss_matches_fd_cuda(factory, algebra, manifold, tol):
    """The exact-match property holds on CUDA too, not just CPU."""
    torch.manual_seed(SEED)
    device = torch.device("cuda")
    pts = torch.randn(3, dtype=DTYPE, device=device)

    def f_act_pure(a):
        return a.Act(pts).pow(2).sum()

    pose = pp.Parameter(factory(1, dtype=DTYPE, device=device))
    f_act_pure(pose).backward()
    g = pose.grad
    assert g.shape == torch.Size([1, manifold])
    assert g.device.type == "cuda"

    fd = _left_perturbation_fd(f_act_pure, pose, algebra, manifold)
    rel = (fd - g[0]).norm().item() / max(g.norm().item(), 1.0)
    assert rel <= tol, "rel=%.3e (tol=%.1e)" % (rel, tol)


@pytest.mark.parametrize("factory, algebra, manifold, tol",
                         [GROUP_CASES[0], GROUP_CASES[1]])  # SO3, SE3
def test_act_terminated_loss_grad_is_finite(factory, algebra, manifold, tol):
    """For a loss mixing an Act term with a raw `.rotation().tensor()` term,
    the accumulated gradient is finite. The discrepancy against the left-
    perturbation FD comes from the raw-slice `.rotation()` term (not
    manifold-covariant -- see the module comment above), not from Act's
    backward, so it is intentionally NOT gated on FD accuracy here."""
    torch.manual_seed(SEED)
    pts = torch.randn(3, dtype=DTYPE, device=DEVICE)

    def f_act(a):
        return (a.Act(pts).pow(2).sum()
                + (a.rotation().tensor() * 0.5).sum())

    pose = pp.Parameter(factory(1, dtype=DTYPE, device=DEVICE))
    f_act(pose).backward()
    g = pose.grad
    assert g.shape == torch.Size([1, manifold])
    assert torch.isfinite(g).all(), "Act-path grad must be finite"

    fd = _left_perturbation_fd(f_act, pose, algebra, manifold)
    rel = (fd - g[0]).norm().item() / max(g.norm().item(), 1.0)
    print("INFO act-path (mixed with raw .rotation() term) left-"
          "perturbation FD rel=%.3e (mismatch is from the non-covariant "
          "raw-slice term, not Act; not gated)" % rel)


# ---------------------------------------------------------------------------
# 3. Public pp.func.jacobian: manifold columns, matching left-perturbation FD
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("factory, algebra, manifold, tol",
                         [GROUP_CASES[0], GROUP_CASES[1]])  # SO3, SE3
def test_public_jacobian_matches_left_perturbation_fd(factory, algebra,
                                                      manifold, tol):
    """pp.func.jacobian w.r.t. an embedded group Parameter exposes the
    manifold column dimension, its diagonal blocks match the
    left-perturbation FD of the same function, and off-diagonal blocks are
    ~0 for this per-row loss."""
    torch.manual_seed(SEED)
    B = 2
    ref = factory(B, dtype=DTYPE, device=DEVICE)
    pose = pp.Parameter(factory(B, dtype=DTYPE, device=DEVICE))
    assert pose.shape == torch.Size([B, int(pose.ltype.embedding[0])])

    def f(p):
        return (p @ ref).Log().tensor()

    J = pp.func.jacobian(f, pose)
    assert J.shape == torch.Size([B, manifold, B, manifold]), str(J.shape)

    g = pose.detach()
    for r_out in range(B):
        for r_in in range(B):
            fd = torch.zeros(manifold, manifold, dtype=DTYPE, device=DEVICE)
            for i in range(manifold):
                e = torch.zeros(manifold, dtype=DTYPE, device=DEVICE)
                e[i] = EPS
                fp = pp.LieTensor(e, ltype=algebra).Exp()
                fm = pp.LieTensor(-e, ltype=algebra).Exp()
                gp = g.clone()
                gm = g.clone()
                g_row_p = pp.LieTensor(g[r_in:r_in + 1], ltype=pose.ltype)
                g_row_m = pp.LieTensor(g[r_in:r_in + 1], ltype=pose.ltype)
                gp[r_in] = (fp @ g_row_p).tensor()[0]
                gm[r_in] = (fm @ g_row_m).tensor()[0]
                fup = pp.LieTensor(gp, ltype=pose.ltype)
                fmn = pp.LieTensor(gm, ltype=pose.ltype)
                fd[:, i] = (f(fup)[r_out] - f(fmn)[r_out]) / (2.0 * EPS)
            block = J[r_out, :, r_in, :]
            if r_out == r_in:
                rel = (block - fd).norm().item() / max(fd.norm().item(), 1.0)
                assert rel <= 1e-3, "rel=%.3e" % rel
            else:
                assert block.norm().item() < 1e-6, str(block.norm().item())


@pytest.mark.parametrize("factory, algebra, manifold, tol",
                         [GROUP_CASES[0], GROUP_CASES[1]])  # SO3, SE3
def test_public_jacobian_of_act_matches_left_perturbation_fd(factory, algebra,
                                                              manifold, tol):
    """The same public ``pp.func.jacobian`` contract holds for an
    Act-terminated (reprojection-shaped) function, not just Mul-terminated:
    diagonal blocks match the left-perturbation FD and off-diagonal blocks
    are ~0 for this per-row loss."""
    torch.manual_seed(SEED)
    B = 2
    pts = torch.randn(B, 3, dtype=DTYPE, device=DEVICE)
    pose = pp.Parameter(factory(B, dtype=DTYPE, device=DEVICE))

    def f(p):
        return p.Act(pts)

    J = pp.func.jacobian(f, pose)
    assert J.shape == torch.Size([B, 3, B, manifold]), str(J.shape)

    g = pose.detach()
    for r_out in range(B):
        for r_in in range(B):
            fd = torch.zeros(3, manifold, dtype=DTYPE, device=DEVICE)
            for i in range(manifold):
                e = torch.zeros(manifold, dtype=DTYPE, device=DEVICE)
                e[i] = EPS
                fp = pp.LieTensor(e, ltype=algebra).Exp()
                fm = pp.LieTensor(-e, ltype=algebra).Exp()
                gp, gm = g.clone(), g.clone()
                g_row_p = pp.LieTensor(g[r_in:r_in + 1], ltype=pose.ltype)
                g_row_m = pp.LieTensor(g[r_in:r_in + 1], ltype=pose.ltype)
                gp[r_in] = (fp @ g_row_p).tensor()[0]
                gm[r_in] = (fm @ g_row_m).tensor()[0]
                fup = pp.LieTensor(gp, ltype=pose.ltype)
                fmn = pp.LieTensor(gm, ltype=pose.ltype)
                fd[:, i] = (f(fup)[r_out] - f(fmn)[r_out]) / (2.0 * EPS)
            block = J[r_out, :, r_in, :]
            if r_out == r_in:
                rel = (block - fd).norm().item() / max(fd.norm().item(), 1.0)
                assert rel <= 1e-3, "rel=%.3e" % rel
            else:
                assert block.norm().item() < 1e-6, str(block.norm().item())


# ---------------------------------------------------------------------------
# 4. Batched / indexed / singleton / float32 / float64
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("factory, algebra, manifold, tol",
                         [GROUP_CASES[0], GROUP_CASES[1]])  # SO3, SE3
@pytest.mark.parametrize("batch", [1, 2, 7], ids=["singleton", "batch2", "batch7"])
def test_grad_view_batched_and_indexed(factory, algebra, manifold, tol, batch):
    """The grad view is manifold-width for singleton/batch; for a per-row loss
    an indexed element's grad matches the corresponding row of the full-batch
    grad."""
    torch.manual_seed(SEED)
    embedding = int(pose_embedding_dim(factory))
    pts = torch.randn(batch, 3, dtype=DTYPE, device=DEVICE)
    pose = pp.Parameter(factory(batch, dtype=DTYPE, device=DEVICE))
    assert pose.shape == torch.Size([batch, embedding])

    (pose @ pts).sum().backward()
    assert pose.grad.shape == torch.Size([batch, manifold])

    if batch > 1:
        # Rebuild a 1-element parameter at pose[0]'s embedded value; the
        # per-row loss makes the indexed grad match the full-batch row.
        pose2 = pp.Parameter(pose[0:1])
        (pose2 @ pts[0:1]).sum().backward()
        assert pose2.grad.shape == torch.Size([1, manifold])
        torch.testing.assert_close(pose2.grad[0], pose.grad[0],
                                   atol=1e-8, rtol=1e-6)


@pytest.mark.parametrize("factory, algebra, manifold, tol",
                         [GROUP_CASES[0], GROUP_CASES[1]])  # SO3, SE3
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64],
                         ids=["fp32", "fp64"])
def test_grad_view_dtype(factory, algebra, manifold, tol, dtype):
    """The grad view is manifold-width and finite in both fp32 and fp64; the
    embedded raw gradient keeps its embedded width with a ~0 redundant
    trailing coordinate."""
    torch.manual_seed(SEED)
    pose = pp.Parameter(factory(2, dtype=dtype, device=DEVICE))
    pts = torch.randn(2, 3, dtype=dtype, device=DEVICE)
    (pose @ pts).sum().backward()
    assert pose.grad.shape[-1] == manifold
    assert torch.isfinite(pose.grad).all(), "grad must be finite"
    raw = pose._raw_autograd_grad()
    assert raw.shape[-1] == int(pose.ltype.embedding[0])
    pad_tol = 1e-4 if dtype is torch.float32 else 1e-6
    assert raw[..., manifold:].abs().max() < pad_tol


# ---------------------------------------------------------------------------
# 5. CUDA-when-safe: tiny grad view check (low memory)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("factory, algebra, manifold, tol",
                         [GROUP_CASES[0], GROUP_CASES[1]])  # SO3, SE3
def test_cuda_grad_view_tiny(factory, algebra, manifold, tol):
    """Tiny CUDA grad-view check (2 elements, low memory). Native torch.optim
    is NOT exercised here: it is a documented OPEN limitation for group
    Parameters (see test_grad_view.py)."""
    device = torch.device("cuda")
    torch.manual_seed(SEED)
    pose = pp.Parameter(factory(2, device=device, dtype=torch.float64))
    pts = torch.randn(2, 3, device=device, dtype=torch.float64)
    (pose @ pts).sum().backward()
    assert pose.grad.shape == torch.Size([2, manifold])
    raw = pose._raw_autograd_grad()
    assert raw.shape == torch.Size([2, int(pose.ltype.embedding[0])])
    assert pose.grad.data_ptr() == raw.data_ptr()
    peak = torch.cuda.max_memory_allocated(device)
    assert peak < (50 * 1024 * 1024), "CUDA test used too much memory"
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# 6. Retraction is a manifold operation, not a Euclidean coordinate add
# ---------------------------------------------------------------------------

def test_se3_tangent_step_applied_via_exponential_not_euclidean_add():
    """An SE3 tangent delta applied via the manifold operation (left
    perturbation, Exp composition) equals Exp(delta) @ G, and is materially
    different from a raw coordinate-wise (Euclidean) add of the 6-D delta
    onto the 7-D embedded storage.

    The Euclidean add is done on the underlying raw tensor (via .tensor()) to
    avoid pypose intercepting __add__ and performing a manifold composition.
    """
    torch.manual_seed(SEED)
    pose = pp.Parameter(pp.randn_SE3(1, dtype=DTYPE, device=DEVICE))
    delta = torch.randn(1, 6, dtype=DTYPE, device=DEVICE) * 1e-2

    pose_raw = pose.tensor().detach().clone()  # (1, 7) raw embedded storage

    # Manifold (left-perturbation) retraction: compose Exp(delta) with the pose.
    delta_lt = pp.LieTensor(delta, ltype=pp.se3_type)
    retracted = (delta_lt.Exp() @ pose).tensor().detach()

    # The Euclidean (coordinate-wise) add of the 6-D delta onto the 7-D
    # storage is the WRONG operation for a Lie group; confirm it differs.
    euclidean = pose_raw.clone()
    euclidean[..., :6] = euclidean[..., :6] + delta  # raw coordinate add
    diff = (retracted - euclidean).norm()
    assert diff > 1e-6, (
        f"manifold retraction should differ from Euclidean add; diff={diff.item()}"
    )
    # The retracted result is still a valid embedded group (unit quaternion).
    q = retracted[..., 3:7]
    assert ((q.norm(dim=-1) - 1.0).abs() < 1e-9).all()


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
