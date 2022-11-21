import torch as torch
import torch.nn as nn
import pypose as pp
# from torch.autograd.functional import jacobian

from pypose.module.dynamics import System
import math
import numpy as np
import matplotlib.pyplot as plt


logcost = torch.Tensor([1.0])
err = torch.Tensor([1.1])
candidate = torch.vstack((logcost, err))
filter = torch.hstack( (candidate - 1.0, candidate+2.0) )

if torch.any( torch.all(candidate>=filter, 0) ):
    failed=True
    print('here')
    # break                    
else:
    idx=torch.all(candidate<=filter,0) #todo: check
    filter = filter[:,~idx] #todo: check
    filter=torch.hstack((filter,candidate))
    print('filter', filter)