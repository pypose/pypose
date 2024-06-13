import pypose as pp
import torch as torch

if __name__ == "__main__":
        n_batch = 4
        T = 5        
        gx = torch.tile(torch.randn(2, 3), (n_batch, T, 1, 1))
        gu = torch.tile(torch.randn(2, 2), (n_batch, T, 1, 1))
        g = torch.randn(n_batch, T, 2)
        state = torch.randn(n_batch, T, 3)
        input = torch.randn(n_batch, T, 2)
        lincon = pp.module.LinCon(gx,gu,g)
        print(lincon(state, input).shape)
        print(lincon.gx.shape)