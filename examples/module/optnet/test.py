import torch
import numpy as np


a = np.array([[1,1,1,0,0,0],[1,1,1,0,0,0]]) 
print(np.shape(a))

b = a.repeat([2,1],axis=0)
print(b)
print(np.shape(b))


# x = [[1, 0, 0, -1, 0, 0], [0, 1, 0, -1, -1, 0], [0, 0, 1, -1, -1, -1]]

# ad = [1,1,1,0,0,0]


# trainY  = torch.ones(10, 6)

# print(trainY)


