import pypose as pp
import torch as torch

if __name__ == "__main__":
    Q = torch.eye(3, 3)
    S = torch.zeros(3, 2)
    R = torch.eye(2, 2)
    c = torch.zeros(1, 1)
    state = torch.randn(1, 3)
    input = torch.randn(1, 2)
    quadcost = pp.module.QuadCost(Q, R, S, c)
    print(state, input)
    quadcost.set_refpoint(state=state, input=input)
    print(quadcost.cx.size())
    print(quadcost.cxx.size())