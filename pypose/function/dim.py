import torch

def excludeBatch(inp, type=1):
    r'''
    Excludes the extra dims in the jacobian of batched tensor
    '''
    if inp.ndim <= 4: # non-batch case
        # for cost cx  (1,1,  1,ns)
        # for cost cxx (1,ns, 1,ns)
        # for constraint gx (ncon,1,  1,ns)
        out = inp.squeeze(-2)
        if out.shape[0]==1:
            out = out.squeeze(0)
        else:
            out = out.squeeze(1)

    else:
        B = inp.shape[-3:-1]
        if type == 1: # zero dim per sample
            out = torch.zeros(inp.shape[-3:], dtype=inp.dtype, device=inp.device)
            for i in range(B[0]): #todo: compatible with non-batch case
                for j in range(B[1]):
                    out[i,j,:] = inp[i,j,i,j,:]
        if type == 2: # vector per sample
            out = torch.zeros(inp.shape[:3]+(inp.shape[-1:]), dtype=inp.dtype, device=inp.device)
            for i in range(B[0]):
                for j in range(B[1]):
                    out[i,j,:,:] = inp[i,j,:,i,j,:]
    return out
