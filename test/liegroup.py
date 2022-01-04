
import torch
import pypose
from pypose.liegroup.lietorch import SO3

# print(lietorch.__version__)
torch.manual_seed(0)
torch.cuda.manual_seed(0)

phi = torch.randn(8000, 3, device='cuda', requires_grad=True)
R = SO3.exp(phi)

# relative rotation matrix, SO3 ^ {8000 x 8000}
dR = R[:,None].inv() * R[None,:]

# 8000x8000 matrix of angles
ang = dR.log().norm(dim=-1)

# backpropogation in tangent space
loss = ang.sum()
loss.backward()
print(R.data.grad)
print(phi.grad)
