import torch as torch
from pypose.module.cost import Cost

# Create class for non-quadratic cost
class NonQuadCost(Cost):
    def __init__(self):
        super(NonQuadCost, self).__init__()

    def cost(self, state, input):
        return torch.sum(state**3,dim=-1) + torch.sum(input**4,dim=-1) \
                + torch.sum(state * input,dim=-1) # did not consider batch here


if __name__ == "__main__":
    n_batch = 2
    T = 1
    state = torch.randn(n_batch, T, 3)
    print(state)
    input = torch.randn(n_batch, T, 3)
    # Create cost object
    nonQuadCost = NonQuadCost()
    cost_value = nonQuadCost(state, input)
    print('cost_value', cost_value.size())
    # 1st, 2nd order partial derivatives at current state and input
    jacob_state, jacob_input = state, input
    nonQuadCost.set_refpoint(state=jacob_state, input=jacob_input)
    print('cx', nonQuadCost.cx.size(), torch.linalg.norm(nonQuadCost.cx - (3*state**2 + input)))
    print('cu', nonQuadCost.cu.size(), torch.linalg.norm(nonQuadCost.cu - (4*input**3 + state)))

    print('cxx', nonQuadCost.cxx.size(), nonQuadCost.cxx, '?==?')
    print('cxu', nonQuadCost.cxu.size(), nonQuadCost.cxu, '?==?')
    print('cux', nonQuadCost.cux.size(), nonQuadCost.cux, '?==?')
    print('cuu', nonQuadCost.cuu.size(), nonQuadCost.cuu, '?==?')
    print('c', nonQuadCost.c.size())

    cx, cu, cxx, cxu, cux, cuu, c = nonQuadCost.cx, nonQuadCost.cu, nonQuadCost.cxx, nonQuadCost.cxu, nonQuadCost.cux, nonQuadCost.cuu, nonQuadCost.c

    