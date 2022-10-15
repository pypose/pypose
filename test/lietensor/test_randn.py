import time
import copy
import torch
import random
import warnings
import torch.linalg
import pypose as pp
from torch import nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_randn():

    x = pp.SO3([0, 0, 0, 1])
    pp.randn_like(x)
    pp.randn_SO3(x.lshape, dtype=x.dtype, layout=x.layout, device=x.device)
    a = pp.randn_Sim3(2, sigma=(1.0, 1.0, 2.0, 1.0, 2.0))
    pp.randn_like(a)
    pp.randn_so3(2, sigma=0.1, requires_grad=True, dtype=torch.float64)
    pp.randn_so3((2, 2), sigma=0.1, requires_grad=True, dtype=torch.float64)
    pp.randn_so3(2, 2, sigma=0.1, requires_grad=True, dtype=torch.float64)
    pp.randn_SO3(2, sigma=0.1, requires_grad=True, dtype=torch.float64)
    pp.randn_se3(2, sigma=(1.0, 0.5))
    pp.randn_se3(2, sigma=(1.0, 2.0, 3.0, 0.5))
    pp.randn_SE3(2, sigma=(1.0, 2.0)) 
    pp.randn_SE3((2, 2), sigma=(1.0, 2.0), requires_grad=True, dtype=torch.float64)
    pp.randn_SE3(2, 2, sigma=(1.0, 2.0), requires_grad=True, dtype=torch.float64)
    pp.randn_SE3(2, sigma=(1.0, 1.5, 2.0, 2.0))
    pp.randn_sim3(2, sigma=(1.0, 1.0, 2.0)) 
    pp.randn_sim3(2, sigma=(1.0, 1.0, 2.0, 1.0, 2.0))
    pp.randn_Sim3(2, sigma=(1.0, 1.0, 2.0))
    pp.randn_Sim3((2, 2, 3), sigma=(1.0, 1.0, 2.0), requires_grad=True, dtype=torch.float64)
    pp.randn_Sim3(2, 2, 3, sigma=(1.0, 1.0, 2.0), requires_grad=True, dtype=torch.float64)
    pp.randn_Sim3(2, sigma=(1.0, 1.0, 2.0, 1.0, 2.0))
    pp.randn_rxso3(2, sigma=(1.0, 2.0))
    pp.randn_RxSO3(2, sigma=(1.0, 2.0))



if __name__ == '__main__':
    test_randn()
