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

A = torch.randn((5,4,4))
B = torch.randn((5,4,2))
C = torch.randn((5,3,4))
D = torch.randn((5,3,2))
c1 = torch.randn((5,1,4))
c2 = torch.randn((5,1,3))
x = torch.randn((5,1,4))
u = torch.randn((5,1,2))

lti = pp.module.LTI(A, B, C, D, c1, c2)

#print(A,B,C,D,c1,c2,x,u)    
#The user can implement this line to print each parameter for comparison.

print(lti(x,u))

"""
    The most general case that all parameters are in the batch. 
    The user could change the corresponding values according to the actual physical system and directions above.
"""
