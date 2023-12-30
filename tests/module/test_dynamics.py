import math
import pypose as pp
import torch as torch


def test_dynamics_cartpole():
    """
    Manually generate a trajectory for a forced cart-pole system and
    compare the trajectory and linearization.
    The reference data is originally obtained from the cartpole case
    in the examples folder.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # The reference data
    state_ref = torch.tensor([
        [0.000000000000000000e+00, 0.000000000000000000e+00, 3.141592653589793116e+00, 0.000000000000000000e+00],
        [0.000000000000000000e+00, 4.004595033211845673e-18, 3.141592653589793116e+00, 8.009190066423691346e-18],
        [4.004595033211845793e-20, 4.444370253226554788e-06, 3.141592653589793116e+00, 2.222185126625291286e-06],
        [4.444370253230559533e-08, 1.333266620835869948e-05, 3.141592675811644586e+00, 6.666333104197371212e-06],
        [1.777703646158925914e-07, 2.666327252216060868e-05, 3.141592742474975442e+00, 1.333054627928972545e-05]],
        device=device)
    A_ref = torch.tensor([[
        [1.0, 0.01, 0.0, 0.0],
        [0.0, 1.0, -0.03270001922006555, -3.4808966152769563e-07],
        [0.0, 0.0, 1.0, 0.01],
        [0.0, 0.0, -0.06539991042629863, 0.9999998259560131]]],
        device=device)
    B_ref = torch.tensor([[0.0], [0.00044444299419410527], [0.0], [0.00022222042025532573]], device=device)
    C_ref = torch.eye(4, device=device)
    D_ref = torch.zeros((4,1), device=device)
    c1_ref = torch.tensor([0.0, 0.10273013763290852, 0.0, 0.20545987669936888], device=device)
    c2_ref = torch.zeros((4,), device=device)

    # The class
    class CartPole(pp.module.NLS):
        def __init__(self):
            super().__init__()
            self._tau = 0.01
            self._length = 1.5
            self._cartmass = 20.0
            self._polemass = 10.0
            self._gravity = 9.81
            self._polemassLength = self._polemass * self._length
            self._totalMass = self._cartmass + self._polemass

        def state_transition(self, state, input, t=None):
            x = state[..., 0]
            xDot = state[..., 1]
            theta = state[..., 2]
            thetaDot = state[..., 3]
            force = input[..., 0]
            costheta = torch.cos(theta)
            sintheta = torch.sin(theta)

            temp = (
                force + self._polemassLength * thetaDot**2 * sintheta
            ) / self._totalMass
            thetaAcc = (self._gravity * sintheta - costheta * temp) / (
                self._length * (4.0 / 3.0 - self._polemass * costheta**2 / self._totalMass)
            )
            xAcc = temp - self._polemassLength * thetaAcc * costheta / self._totalMass

            _dstate = torch.stack((xDot, xAcc, thetaDot, thetaAcc), dim=-1)
            return state + torch.mul(_dstate, self._tau)

        def observation(self, state, input, t=None):
            return state

    # Time and input
    dt = 0.01
    N  = 1000
    time  = torch.arange(0, N + 1, device=device) * dt
    input = torch.sin(time)
    # Initial state
    state = torch.tensor([0, 0, math.pi, 0], device=device)

    # Create dynamics solver object
    cartPoleSolver = CartPole()

    # Calculate trajectory
    state_all = torch.zeros(N + 1, 4, device=device)
    state_all[0, :] = state
    for i in range(N):
        state_all[i + 1], _ = cartPoleSolver.forward(state_all[i], input[i])

    assert torch.allclose(state_ref, state_all[:5], rtol=1e-2)

    # Jacobian computation - Find jacobians at the last step
    jacob_state, jacob_input = state_all[-1], input[-1]
    cartPoleSolver.set_refpoint(state=jacob_state, input=jacob_input.unsqueeze(0), t=time[-1])

    assert torch.allclose(A_ref, cartPoleSolver.A)
    assert torch.allclose(B_ref, cartPoleSolver.B)
    assert torch.allclose(C_ref, cartPoleSolver.C)
    assert torch.allclose(D_ref, cartPoleSolver.D)
    assert torch.allclose(c1_ref, cartPoleSolver.c1)
    assert torch.allclose(c2_ref, cartPoleSolver.c2)


def test_dynamics_floquet():
    """
    Manually generate a trajectory for a floquet system (which is time-varying)
    and compare the trajectory and linearization against alternative solutions.
    This is for testing a time-varying system.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    N     = 80                    # Number of time steps
    idx   = 5                     # The step to compute jacobians
    time  = torch.arange(0, N + 1, device=device)  # Time steps
    state = torch.tensor([1, 1], device=device)  # Initial state

    # The reference data
    def f(x, t):
        cc = torch.cos(2 * math.pi * t / 100)
        ss = torch.sin(2 * math.pi *t / 100)
        ff = torch.atleast_1d(torch.sin(2 * math.pi * t / 50))
        A = torch.tensor([[   1., cc/10],
                        [cc/10,    1.]], device=device)
        B = torch.tensor([[ss],
                        [1.]], device=device)
        return A.matmul(x) + B.matmul(ff), A, B

    state_ref = torch.zeros(N + 1, 2, device=device)
    state_ref[0] = state
    for i in range(N):
        state_ref[i + 1], _, _ = f(state_ref[i], time[i])
    obser_ref = state_ref[:-1] + time[:-1].reshape(-1, 1)

    _, A0_N, B0_N = f(torch.tensor([0., 0.], device=device), torch.tensor(N, device=device))
    _, A0_i, B0_i = f(torch.tensor([0., 0.], device=device), torch.tensor(idx, device=device))
    c2_N = torch.ones(2, device=device) * N
    c2_i = torch.ones(2, device=device) * idx
    C0 = torch.eye(2, device=device)
    D0 = torch.zeros(2, 1, device=device)
    c1 = torch.zeros(2, device=device)

    # The class
    class Floquet(pp.module.NLS):
        def __init__(self):
            super(Floquet, self).__init__()

        def state_transition(self, state, input, t):
            cc = torch.cos(2*math.pi*t/100)
            ss = torch.sin(2*math.pi*t/100)
            A = torch.tensor([[   1., cc/10],
                            [cc/10,    1.]], device=device)
            B = torch.tensor([[ss],
                            [1.]], device=device)
            return state.matmul(A) + B.matmul(input)

        def observation(self, state, input, t):
            return state + t

    # Input
    input = torch.sin(2 * math.pi * time / 50)

    # Create dynamics solver object
    solver = Floquet().to(device)

    # Calculate trajectory
    state_all = torch.zeros(N + 1, 2, device=device)
    state_all[0] = state
    obser_all = torch.zeros(N, 2, device=device)

    for i in range(N):
        state_all[i + 1], obser_all[i] = solver(state_all[i], input[i])

    assert torch.allclose(state_all, state_ref, atol=1e-5)
    assert torch.allclose(obser_all, obser_ref)

    # # For debugging
    # import matplotlib.pyplot as plt
    # f, ax = plt.subplots(nrows=4, sharex=True)
    # for _i in range(2):
    #     ax[_i].plot(time, state_all[:,_i], label='pp')
    #     ax[_i].plot(time, state_ref[:,_i], label='np')
    #     ax[_i].set_ylabel(f'State {_i}')
    # for _i in range(2):
    #     ax[_i+2].plot(time[:-1], obser_all[:,_i], label='pp')
    #     ax[_i+2].plot(time[:-1], obser_ref[:,_i], label='np')
    #     ax[_i+2].set_ylabel(f'Observation {_i}')
    # ax[-1].set_xlabel('time')
    # ax[-1].legend()
    # plt.show()

    # Jacobian computation - at the last step
    # Note for c1, the values are supposed to be zero, but due to numerical
    # errors the values can be ~ 1e-7, and hence we increase the atol
    # Same story below
    solver.set_refpoint()
    assert torch.allclose(A0_N, solver.A)
    assert torch.allclose(B0_N, solver.B)
    assert torch.allclose(C0, solver.C)
    assert torch.allclose(D0, solver.D)
    assert torch.allclose(c1, solver.c1, atol=1e-6)
    assert torch.allclose(c2_N, solver.c2)

    # Jacobian computation - at the step idx
    solver.set_refpoint(state=state_all[idx], input=input[idx], t=time[idx])
    assert torch.allclose(A0_i, solver.A)
    assert torch.allclose(B0_i, solver.B)
    assert torch.allclose(C0, solver.C)
    assert torch.allclose(D0, solver.D)
    assert torch.allclose(c1, solver.c1, atol=1e-6)
    assert torch.allclose(c2_i, solver.c2)


def test_dynamics_lti():

    """
    For a System with p inputs, q outputs, and n state variables,
    A, B, C, D are n*n n*p q*n and q*p constant matrices.
    N: channels

    A = torch.randn(N, n, n)
    B = torch.randn(N, n, p)
    C = torch.randn(N, q, n)
    D = torch.randn(N, q, p)
    c1 = torch.randn(N, 1, n)
    c2 = torch.randn(N, 1, q)
    state = torch.randn(N, 1, n)
    input = torch.randn(N, 1, p)
    """

    # The most general case that all parameters are in the batch.
    # The user could change the corresponding values according to the actual physical system and directions above.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    A_1 = torch.randn(5, 4, 4, device=device)
    B_1 = torch.randn(5, 4, 2, device=device)
    C_1 = torch.randn(5, 3, 4, device=device)
    D_1 = torch.randn(5, 3, 2, device=device)
    c1_1 = torch.randn(5, 4, device=device)
    c2_1 = torch.randn(5, 3, device=device)
    state_1 = torch.randn(5, 4, device=device)
    input_1 = torch.randn(5, 2, device=device)

    lti_1 = pp.module.LTI(A_1, B_1, C_1, D_1, c1_1, c2_1).to(device)

    # The user can implement this line to print each parameter for comparison.

    z_1, y_1 = lti_1(state_1,input_1)

    z_1_ref = pp.bmv(A_1, state_1) + pp.bmv(B_1, input_1) + c1_1

    y_1_ref = pp.bmv(C_1, state_1) + pp.bmv(D_1, input_1) + c2_1

    assert torch.allclose(z_1, z_1_ref)
    assert torch.allclose(y_1, y_1_ref)


    #In this example, A, B, C, D, c1, c2 are single inputs, state and input are in a batch.

    A_2 = torch.randn(4, 4, device=device)
    B_2 = torch.randn(4, 2, device=device)
    C_2 = torch.randn(3, 4, device=device)
    D_2 = torch.randn(3, 2, device=device)
    c1_2 = torch.randn(4, device=device)
    c2_2 = torch.randn(3, device=device)
    state_2 = torch.randn(5, 4, device=device)
    input_2 = torch.randn(5, 2, device=device)

    lti_2 = pp.module.LTI(A_2, B_2, C_2, D_2, c1_2, c2_2).to(device)

    z_2, y_2 = lti_2(state_2, input_2)

    z_2_ref = pp.bmv(A_2, state_2) + pp.bmv(B_2, input_2) + c1_2
    y_2_ref = pp.bmv(C_2, state_2) + pp.bmv(D_2, input_2) + c2_2

    assert torch.allclose(z_2, z_2_ref)
    assert torch.allclose(y_2, y_2_ref)


    # In this example, all parameters are single inputs.

    A_3 = torch.randn(4, 4, device=device)
    B_3 = torch.randn(4, 2, device=device)
    C_3 = torch.randn(3, 4, device=device)
    D_3 = torch.randn(3, 2, device=device)
    c1_3 = torch.randn(4, device=device)
    c2_3 = torch.randn(3, device=device)
    state_3 = torch.randn(4, device=device)
    input_3 = torch.randn(2, device=device)

    lti_3 = pp.module.LTI(A_3, B_3, C_3, D_3, c1_3, c2_3).to(device)

    z_3, y_3 = lti_3(state_3, input_3)

    z_3_ref = A_3.mv(state_3) + B_3.mv(input_3) + c1_3
    y_3_ref = C_3.mv(state_3) + D_3.mv(input_3) + c2_3

    assert torch.allclose(z_3, z_3_ref)
    assert torch.allclose(y_3, y_3_ref)


if __name__ == '__main__':
    test_dynamics_cartpole()
    test_dynamics_floquet()
    test_dynamics_lti()
