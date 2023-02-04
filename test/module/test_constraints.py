import pypose as pp
import torch as torch

if __name__ == "__main__":
        gx = torch.randn(2, 3)
        gu = torch.randn(2, 2)
        g = torch.randn(2, 1)
        state = torch.randn(1, 3)
        input = torch.randn(1, 2)
        lincon = pp.module.LinCon(gx,gu,g)
        print(lincon(state, input))
        print(lincon.gx)