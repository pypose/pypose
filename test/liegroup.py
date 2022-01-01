
import torch
import pypose
from pypose.liegroup.lietorch import SO3

# print(lietorch.__version__)

phi = torch.randn(8000, 3, device='cuda', requires_grad=True)
R = SO3.Exp(phi)

# relative rotation matrix, SO3 ^ {8000 x 8000}
dR = R[:,None].Inv() * R[None,:]

# 8000x8000 matrix of angles
ang = dR.Log().norm(dim=-1)

# backpropogation in tangent space
loss = ang.sum()
loss.backward()
print(R.grad)
print(phi.grad)
