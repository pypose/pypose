import sys
sys.path.append("..")
import torch as torch
import pypose as pp
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

