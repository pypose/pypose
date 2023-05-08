import pypose as pp
from torch import nn
import functorch, functools
import sys, math, warnings
import torch
import numpy as np
np.set_printoptions(precision=3)
from torch.autograd.functional import jacobian

import time #to count time
import gc
import argparse
import os

device = []
log = []
jac_shape = []

def test_pypose_jacobian(batch_size, use_gpu, op = 0):

    rot1 = torch.randn(5)
    
    f = lambda x: x**2
    

    y = f( rot1 )
    I = torch.randn(batch_size, 5)
    I = I.to(device)

    # vectorized gradient computation to match Theseus output
    def get_vjp(v):

        return torch.autograd.grad(y, rot1, v)

    temp = functorch.vmap(get_vjp, in_dims=1)
    J = temp(I)[0].permute(1,0,2)

    print('J.shape:', J.shape)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--use_gpu', type = int, default = 0)
    parser.add_argument('--itrs', type = int, default = 100)
    parser.add_argument('--batch_num', type = int, default = 8)
    parser.add_argument('--operations', type = int, default = 3) 
    args = parser.parse_args()
    
    # convert args to dictionary
    params = vars(args)

    batch_sizes = [10]
    
    itrs = params["itrs"]
    ops = params["operations"]
    use_gpu = params["use_gpu"]
    batch_num = params["batch_num"] # greatest batch_size = 10^(batch_num)
    batch_num = min(batch_num,8)

    for op in range(ops):

        if(use_gpu):
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")


        for itr in range(itrs):
            for i in range( batch_num ):
            
                test_pypose_jacobian(batch_sizes[i], use_gpu, op)

    