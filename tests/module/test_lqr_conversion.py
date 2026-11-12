import pytest
import torch
import pypose as pp


def make_controller(shared_cost=False, parameter_cost=False, mpc=False):
    A = torch.eye(2).unsqueeze(0)
    B = torch.tensor([[[0.], [1.]]])
    system = pp.module.LTI(A, B, A.clone(), torch.zeros_like(B))
    Q = torch.eye(3).repeat(1, 3, 1, 1)
    p = torch.zeros(1, 3, 3)
    if shared_cost:
        Q, p = Q[:, 0], p[:, 0]
    if parameter_cost:
        Q, p = torch.nn.Parameter(Q), torch.nn.Parameter(p)
    if mpc:
        return pp.module.MPC(system, Q, p, 3, stepper=pp.utils.ReduceToBason(steps=2))
    return pp.module.LQR(system, Q, p, 3)


def check_solution(controller, dtype, device="cpu"):
    initial = torch.tensor([[1., 2.]], dtype=dtype, device=device)
    if isinstance(controller, pp.module.MPC):
        states, controls, cost = controller(1, initial)
    else:
        states, controls, cost = controller(initial)
    expected_controls = torch.tensor([[[-1.2], [-0.4], [0.]]], dtype=dtype, device=device)
    torch.testing.assert_close(controls, expected_controls, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(cost, torch.tensor([4.7], dtype=dtype, device=device))
    assert states.dtype == controls.dtype == cost.dtype == dtype
    assert states.device == controls.device == cost.device == initial.device


@pytest.mark.parametrize("shared_cost", [False, True])
def test_lqr_costs_follow_module_dtype_conversion(shared_cost):
    controller = make_controller(shared_cost=shared_cost)
    original_keys = set(controller.state_dict())
    check_solution(controller, torch.float32)
    controller.double()
    check_solution(controller, torch.float64)
    controller.float()
    check_solution(controller, torch.float32)
    assert set(controller.state_dict()) == original_keys
    assert "Q" not in controller.state_dict()
    assert "p" not in controller.state_dict()


def test_lqr_keeps_parameter_costs_registered_after_conversion():
    controller = make_controller(parameter_cost=True).double()
    parameters = dict(controller.named_parameters())
    assert parameters["Q"] is controller.Q
    assert parameters["p"] is controller.p
    assert controller.Q.requires_grad and controller.p.requires_grad
    check_solution(controller, torch.float64)


@pytest.mark.parametrize("shared_cost", [False, True])
def test_mpc_costs_follow_module_dtype_conversion(shared_cost):
    controller = make_controller(shared_cost=shared_cost, mpc=True).double()
    check_solution(controller, torch.float64)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_lqr_costs_follow_module_device_conversion():
    controller = make_controller().cuda()
    check_solution(controller, torch.float32, "cuda")
