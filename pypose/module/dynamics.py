import torch as torch
import torch.nn as nn

class _System(nn.Module):
    def __init__(self,time=False):
        super().__init__()
        if time:
            self.register_buffer('t',torch.zeros(1))
            self.register_forward_hook(self.timeplus1)

    def timeplus1(self,module,inputs,outputs):
        self.t.add_(1)

    def forward(self,state,input):
        state = self.state_transition(state,input)
        return self.observation(state,input)
    
    def state_trasition(self):
        pass

    def observation(self):
        pass
    
    def reset(self,t=0):
        self.t.fill_(0)

class LTI(_System):
    def __init__(self, N, A, B=None, C=None, D=None,time=False):
        super().__init__(time)
        self.register_buffer('A', A) #self.A = A
        self.register_buffer('B', B) #self.B = B
        self.register_buffer('C', C) #self.C = C
        self.register_buffer('D', D) #self.D = D

    def state_transition(self, state, input):
        return self.A @ state + self.B @ input
    
    def observation(self, state, input):
        return self.C @ state + self.D @ input

    def forward(self,x,u):
        x = self.state_transition(u,x)
        y = self.observation(u,x)
        return y
    
    
class LTV(_System):
    def __init__(self, N, A, B=None, C=None, D=None,time=False):
        super().__init__(time)
        self.register_buffer('A', A) #self.A = A
        self.register_buffer('B', B) #self.B = B
        self.register_buffer('C', C) #self.C = C
        self.register_buffer('D', D) #self.D = D
        if time:
            A = A*self.t
            B = B*self.t
            C = C*self.t
            D = D*self.t

    def state_transition(self, state, input):
        return self.A @ state + self.B @ input
    
    def observation(self, state, input):
        return self.C @ state + self.D @ input

    def forward(self,x,u):
        x = self.state_transition(u,x)
        y = self.observation(u,x)
        return y
    
    
"""
class _System(nn.Module):
    def __init__(self, A, B, C, D,time = Fasle):
        if time: 
            self.register_buffer('t',torch.zeros(1)) 
            self.register_forward_hook(self.timeplus1) 

        super(_System, self).__init__()

        assert A.ndimension() == 2
        assert B.ndimension() == 2
        assert A.ndimension() == 2
        assert B.ndimension() == 2
        
        self.A = A
        self.B = B
        self.C = C
        self.D = D

    def timeplus1(self): 
        self.t.add_(1) 

    def forward(self, x, u):
        if not isinstance(x, Variable) and isinstance(self.A, Variable):
            A = self.A.data
            B = self.B.data
            c = self.C.data
            d = self.D.data
        else:
            A = self.A
            B = self.B
            c = self.C
            d = self.D

        x_dim, u_dim = x.ndimension(), u.ndimension()
        if x_dim == 1:
            x = x.unsqueeze(0)
        if u_dim == 1:
            u = u.unsqueeze(0)

        z = x.mm(A.t()) + u.mm(B.t())
        y = x.mm(C.t()) + u.mm(D.t())

        if x_dim == 1:
            z = z.squeeze(0)
            y = y.squeeze(0)

        return z, y

    """
