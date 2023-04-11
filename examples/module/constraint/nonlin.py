import torch as torch
from pypose.module.constraint import Constraint

# Create class for non-quadratic cost
class NonLinConstraint(Constraint):
    def __init__(self):
        super(NonLinConstraint, self).__init__()

    def constraint(self, state, input):
        return torch.sum(state**3,dim=-1) + torch.sum(input**4,dim=-1)


if __name__ == "__main__":
    n_batch = 2
    T = 4
    state = torch.randn(n_batch, T, 3)
    input = torch.randn(n_batch, T, 3)
    # Create cost object
    nonLinConstraint = NonLinConstraint()
    constraint_value = nonLinConstraint(state, input)
    print('constraint_value', constraint_value.size())
    # 1st, 2nd order partial derivatives at current state and input
    jacob_state, jacob_input = state, input
    nonLinConstraint.set_refpoint(state=jacob_state, input=jacob_input)
    print('gx', nonLinConstraint.gx.size(), torch.linalg.norm(nonLinConstraint.gx - (3*state**2)))
    print('gu', nonLinConstraint.gu.size(), torch.linalg.norm(nonLinConstraint.gu - (4*input**3)))

    