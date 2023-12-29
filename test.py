
import torch
class Universal():
    def __init__(self):
        super().__init__()
        self.A=torch.randn(3,3,requires_grad=True)
        self.B=torch.randn(3,2,requires_grad=True)

    def state_transition(self, state, input, T):
        # Linear State Transition
        A=self.A*state

        next_state = torch.matmul(A, state) + torch.matmul(self.B, input)

        return next_state

def hook_fn(grad):
    print("A Gradient: ", grad)

# Create an instance of Universal
sys = Universal()

# Initialize tensors with requires_grad=True
time = torch.tensor(10.0, requires_grad=True)
state = torch.randn(3, requires_grad=True)
input = torch.randn(2, requires_grad=True)

# Register hooks
state.register_hook(hook_fn)

out = sys.state_transition(state,input,time)
out.sum().backward()
print(out)
