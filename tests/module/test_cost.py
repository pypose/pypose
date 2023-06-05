import pypose as pp
import torch as torch

if __name__ == "__main__":
    n_batch = 4
    T = 5
    Q = torch.tile(torch.eye(3, 3), (n_batch, T, 1, 1))
    S = torch.tile(torch.zeros(3, 2), (n_batch, T, 1, 1))
    R = torch.tile(torch.eye(2, 2), (n_batch, T, 1, 1)) 
    c = torch.zeros(n_batch, T)
    state = torch.randn(n_batch, T, 3)
    input = torch.randn(n_batch, T, 2)
    quadcost = pp.module.QuadCost(Q, R, S, c)
    print(state.shape, input.shape)
    quadcost.set_refpoint(state=state, input=input)
    print(quadcost.cx.size())
    print(quadcost.cu.size())
    print(quadcost.cxx.size()) # set directly
    # print(pp.bdot(state, torch.randn(n_batch,T,3)).shape)