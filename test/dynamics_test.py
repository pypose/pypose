import sys
sys.path.append("..")
import torch as torch
import pypose as pp
import torch.nn as nn

"""
A(n*n), B(n*p), C(q*n), D(q*p)
x(n*1), u(p*1), N: channels
A = torch.randn((n,n))
B = torch.randn((n,p))
C = torch.randn((q,n))
D = torch.randn((q,p))
c1 = torch.randn(N,1,n)
c2 = torch.randn(N,1,q)
x = torch.randn((N,1,n))
u = torch.randn((N,1,p))
"""
"""
A = torch.randn((3,3))
B = torch.randn((3,2))
C = torch.randn((3,3))
D = torch.randn((3,2))
c1 = torch.randn((2,1,3))
c2 = torch.randn((2,1,3))
x = torch.randn((2,1,3))
u = torch.randn((2,1,2))

lti = pp.module.LTI(A, B, C, D, c1, c2)

print(A,B,C,D,c1,c2,x,u)

print(lti(x,u))
"""

"""
A = torch.tensor([[1,2,4],[2,0,5],[1,1,1]])
B = torch.tensor([[3,2],[5,9],[8,0]])
C = torch.tensor([[5,9,8],[4,4,3],[1,0,1],[3,4,5]])
D = torch.tensor([[0,1],[2,3],[3,-3],[-1,-9]])
x = torch.tensor([[[2,5,7]],[[12,0,3]],[[0,2,1]]])
u = torch.tensor([[[4,6]],[[1,1]],[[0,6]]])
c1 = torch.tensor([[[3,4,5]],[[-2,-4,-6]],[[-3,0,9]]])
c2 = torch.tensor([[[4,6,8,0]],[[1,1,1,1]],[[0,6,6,-2]]])

lti = pp.module.LTI(A, B, C, D, c1, c2)

print(lti(x,u))
"""

A = torch.tensor([[[1,2,4],[2,0,5],[1,1,1]],[[4,5,2],[10,0,9],[6,7,2]],[[3,4,5],[7,7,1],[0,9,8]]])
B = torch.tensor([[[3,2],[5,9],[8,0]],[[4,4],[5,8],[2,11]],[[3,12],[0,8],[13,4]]])
C = torch.tensor([[[5,9,8],[4,4,3],[1,0,1],[3,4,5]],[[10,5,3],[11,12,14],[-3,4,-7],[-1,-3,-5]],[[3,3,3],[-1,-1,-2],[0,-6,-9],[10,0,10]]])
D = torch.tensor([[[0,1],[2,3],[3,-3],[-1,-9]],[[0,-5],[11,12],[-9,-8],[8,-8]],[[-4,-6],[10,0],[15,-2],[0,-4]]])
x = torch.tensor([[[2,5,7]],[[12,0,3]],[[0,2,1]]])
u = torch.tensor([[[4,6]],[[1,1]],[[0,6]]])
c1 = torch.tensor([[[3,4,5]],[[-2,-4,-6]],[[-3,0,9]]])
c2 = torch.tensor([[[4,6,8,0]],[[1,1,1,1]],[[0,6,6,-2]]])

"""
    The most basic situation. 
    The user enters the corresponding values according to the actual physical system and 
    could increase or decrease the number of dimensions.
"""

lti = pp.module.LTI(A, B, C, D, c1, c2)

print(lti(x,u))