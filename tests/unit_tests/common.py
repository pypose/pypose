
import functools

import torch

# Construct a convenient helper function.
torch_equal = functools.partial( 
    torch.testing.assert_close, atol=0, rtol=1e-6 )

def show_delimeter(msg):
    print(f'========== {msg} ==========')
