from pypose.module.cost import Cost
import torch as torch
import numpy as np
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create class for non-quadratic cost
class NonQuadCost(Cost):
    def __init__(self):
        super(NonQuadCost, self).__init__()

    def cost(self, state, input):
        return torch.sum(state**3) + torch.sum(input**4) # did not consider batch here


if __name__ == "__main__":

    # input = torch.tensor([3.], dtype=float)
    # state = torch.tensor([1., 2., np.pi, 4.], dtype=float)
    state = torch.randn(1, 3)
    input = torch.randn(1, 2)
    # Create cost object
    nonQuadCost = NonQuadCost()
    cost_value = nonQuadCost(state, input)
    print('cost_value', cost_value)
    # Jacobian computation - Find jacobians at the last step
    jacob_state, jacob_input = state, input
    nonQuadCost.set_refpoint(state=jacob_state, input=jacob_input)
    print('cx', nonQuadCost.cx.size(), nonQuadCost.cx, '?==?', 3*state**2)
    print('cu', nonQuadCost.cu.size(), nonQuadCost.cu, '?==?', 4*input**3)
    print('cxx', nonQuadCost.cxx.size(), nonQuadCost.cxx, '?==?', 6*torch.diag(state.squeeze(0)))
    print('cxu', nonQuadCost.cxu.size())
    print('cux', nonQuadCost.cux.size())
    print('cuu', nonQuadCost.cuu.size())
    print('c', nonQuadCost.c.size())

    # B = cartPoleSolver.B
    # C = cartPoleSolver.C
    # D = cartPoleSolver.D
    # c1 = cartPoleSolver.c1
    # c2 = cartPoleSolver.c2

    