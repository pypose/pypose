import math
import sys
sys.path.append("..")
import torch as torch
import torch.nn as nn
import pypose as pp
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_dynamics_cartpole():
    """
    Manually generate a trajectory for a forced cart-pole system and
    compare the trajectory and linearization.
    The reference data is originally obtained from the cartpole case
    in the examples folder.
    """

    # The reference data
    state_ref = torch.tensor([
        [0.000000000000000000e+00, 0.000000000000000000e+00, 3.141592653589793116e+00, 0.000000000000000000e+00],
        [0.000000000000000000e+00, 4.004595033211845673e-18, 3.141592653589793116e+00, 8.009190066423691346e-18],
        [4.004595033211845793e-20, 4.444370253226554788e-06, 3.141592653589793116e+00, 2.222185126625291286e-06],
        [4.444370253230559533e-08, 1.333266620835869948e-05, 3.141592675811644586e+00, 6.666333104197371212e-06],
        [1.777703646158925914e-07, 2.666327252216060868e-05, 3.141592742474975442e+00, 1.333054627928972545e-05]],
       dtype=torch.float64)
    A_ref = torch.tensor([[
        [1.0, 0.01, 0.0, 0.0],
        [0.0, 1.0, -0.03270001922006555, -3.4808966152769563e-07],
        [0.0, 0.0, 1.0, 0.01],
        [0.0, 0.0, -0.06539991042629863, 0.9999998259560131]]],
       dtype=torch.float64)
    B_ref = torch.tensor([[0.0], [0.00044444299419410527], [0.0], [0.00022222042025532573]], dtype=torch.float64)
    C_ref = torch.eye(4, dtype=torch.float64)
    D_ref = torch.zeros((4,1), dtype=torch.float64)
    c1_ref = torch.tensor([0.0, 0.10273013763290852, 0.0, 0.20545987669936888], dtype=torch.float64)
    c2_ref = torch.zeros((4,), dtype=torch.float64)

    # The class
    class CartPole(pp.module.System):
        def __init__(self):
            super(CartPole, self).__init__(time=False)
            self._tau = 0.01
            self._length = 1.5
            self._cartmass = 20.0
            self._polemass = 10.0
            self._gravity = 9.81
            self._polemassLength = self._polemass*self._length
            self._totalMass = self._cartmass + self._polemass

        def state_transition(self,state,input):
            x,xDot,theta,thetaDot = state
            force = input.squeeze()
            costheta = torch.cos(theta)
            sintheta = torch.sin(theta)

            temp = (
                force + self._polemassLength * thetaDot**2 * sintheta
            ) / self._totalMass
            thetaAcc = (self._gravity * sintheta - costheta * temp) / (
                self._length * (4.0 / 3.0 - self._polemass * costheta**2 / self._totalMass)
            )
            xAcc = temp - self._polemassLength * thetaAcc * costheta / self._totalMass

            _dstate = torch.stack((xDot,xAcc,thetaDot,thetaAcc))

            return state+torch.mul(_dstate,self._tau)
        
        def observation(self,state,input):
            return state

    # Time and input
    dt = 0.01
    N  = 1000
    time  = torch.arange(0,N+1,dtype=torch.float64) * dt
    input = torch.sin(time)
    # Initial state
    state = torch.tensor([0,0,math.pi,0],dtype=torch.float64)

    # Create dynamics solver object
    cartPoleSolver = CartPole()

    # Calculate trajectory
    state_all = torch.zeros(N+1,4,dtype=torch.float64)
    state_all[0,:] = state
    for i in range(N):
        state_all[i+1], _ = cartPoleSolver.forward(state_all[i],input[i])

    assert torch.allclose(state_ref, state_all[:5])

    # Jacobian computation - Find jacobians at the last step
    jacob_state, jacob_input = state_all[-1,:].T, input[-1]
    cartPoleSolver.set_linearization_point(jacob_state,jacob_input.unsqueeze(0),time[-1])

    assert torch.allclose(A_ref, cartPoleSolver.A)
    assert torch.allclose(B_ref, cartPoleSolver.B)
    assert torch.allclose(C_ref, cartPoleSolver.C)
    assert torch.allclose(D_ref, cartPoleSolver.D)
    assert torch.allclose(c1_ref, cartPoleSolver.c1)
    assert torch.allclose(c2_ref, cartPoleSolver.c2)

    print('Done')

def test_dynamics():

    """
    For a System with p inputs, q outputs, and n state variables,
    A, B, C, D are n*n n*p q*n and q*p constant matrices.
    N: channels

    A = torch.randn((N,n,n))
    B = torch.randn((N,n,p))
    C = torch.randn((N,q,n))
    D = torch.randn((N,q,p))
    c1 = torch.randn(N,1,n)
    c2 = torch.randn(N,1,q)
    state = torch.randn((N,1,n))
    input = torch.randn((N,1,p))
    """

    #The most general case that all parameters are in the batch. The user could change the corresponding values according to the actual physical system and directions above.

    A_1 = torch.randn((5,4,4))
    B_1 = torch.randn((5,4,2))
    C_1 = torch.randn((5,3,4))
    D_1 = torch.randn((5,3,2))
    c1_1 = torch.randn((5,1,4))
    c2_1 = torch.randn((5,1,3))
    state_1 = torch.randn((5,1,4))
    input_1 = torch.randn((5,1,2))

    lti_1 = pp.module.LTI(A_1, B_1, C_1, D_1, c1_1, c2_1)

    #print(A, B, C, D, c1, c2, state, input)    
    #The user can implement this line to print each parameter for comparison.

    print(lti_1(state_1,input_1))


    #In this example, A, B, C, D, c1, c2 are single inputs, state and input are in a batch.

    A_2 = torch.randn((4,4))
    B_2 = torch.randn((4,2))
    C_2 = torch.randn((3,4))
    D_2 = torch.randn((3,2))
    c1_2 = torch.randn((1,4))
    c2_2 = torch.randn((1,3))
    state_2 = torch.randn((5,1,4))
    input_2 = torch.randn((5,1,2))

    lti_2 = pp.module.LTI(A_2, B_2, C_2, D_2, c1_2, c2_2)

    print(lti_2(state_2,input_2))


    #In this example, all parameters are single inputs.

    A_3 = torch.randn((4,4))
    B_3 = torch.randn((4,2))
    C_3 = torch.randn((3,4))
    D_3 = torch.randn((3,2))
    c1_3 = torch.randn((1,4))
    c2_3 = torch.randn((1,3))
    state_3 = torch.randn((1,4))
    input_3 = torch.randn((1,2))

    lti_3 = pp.module.LTI(A_3, B_3, C_3, D_3, c1_3, c2_3)

    print(lti_3(state_3,input_3))

    print('Done')


if __name__ == '__main__':
    test_dynamics()
    test_dynamics_cartpole()

