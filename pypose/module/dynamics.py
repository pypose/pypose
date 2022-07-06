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
