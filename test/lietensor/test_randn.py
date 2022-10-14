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



if __name__ == '__main__':
    test_randn()
