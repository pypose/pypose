from pypose.module.cost import Cost
import torch as torch

# Create class for non-quadratic cost
class NonQuadCost(Cost):
    def __init__(self):
        super(NonQuadCost, self).__init__()

    def cost(self, state, input):
        return torch.sum(state**3) + torch.sum(input**4) \
                + torch.sum(state * input) # did not consider batch here


if __name__ == "__main__":
    state = torch.randn(1, 3)
    input = torch.randn(1, 3)
    # Create cost object
    nonQuadCost = NonQuadCost()
    cost_value = nonQuadCost(state, input)
    print('cost_value', cost_value)
    # 1st, 2nd order partial derivatives at current state and input
    jacob_state, jacob_input = state, input
    nonQuadCost.set_refpoint(state=jacob_state, input=jacob_input)
    print('cx', nonQuadCost.cx.size(), nonQuadCost.cx, '?==?', 3*state**2 + input)
    print('cu', nonQuadCost.cu.size(), nonQuadCost.cu, '?==?', 4*input**3 + state)
    print('cxx', nonQuadCost.cxx.size(), nonQuadCost.cxx, '?==?', 6*torch.diag(state.squeeze(0)))
    print('cxu', nonQuadCost.cxu.size(), nonQuadCost.cxu, '?==?', torch.eye(state.size(-1)) )
    print('cux', nonQuadCost.cux.size(), nonQuadCost.cux, '?==?', torch.eye(state.size(-1)))
    print('cuu', nonQuadCost.cuu.size(), nonQuadCost.cuu, '?==?', 12*torch.diag((input**2).squeeze(0)))
    print('c', nonQuadCost.c.size())

    cx, cu, cxx, cxu, cux, cuu, c = nonQuadCost.cx, nonQuadCost.cu, nonQuadCost.cxx, nonQuadCost.cxu, nonQuadCost.cux, nonQuadCost.cuu, nonQuadCost.c

    