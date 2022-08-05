
import functools

import torch

# Construct a convenient helper function.
torch_equal = functools.partial( 
    torch.testing.assert_close, atol=0, rtol=1e-6 )

torch_equal_rough = functools.partial( 
    torch.testing.assert_close, atol=1e-5, rtol=1e-3 )

def show_delimeter(msg):
    print(f'========== {msg} ==========')
