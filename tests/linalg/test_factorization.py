import os
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import pypose as pp
from torch.distributed._tensor import DeviceMesh

def test_lu(rank, world_size, device_type):

    A = torch.rand(1000, 500)
    dfactori = pp.Dfactori(rank,world_size, device_type)
    device_mesh = DeviceMesh(device_type, torch.arange(world_size))
    L, U = dfactori.factorization(A, device_mesh, 'tslu')
    if rank == 0:
        dresult = torch.matmul(L, U)
        print('input_shape:', dresult.shape, 'error:', torch.sum(dresult- A))


def init_process(rank_id, size, device_type, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    # if use cuda, backend = 'nccl'
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12345'
    dist.init_process_group(backend, rank=rank_id, world_size=size)
    fn(rank_id, size, device_type=device_type)


if __name__ == "__main__":
    world_size = 4
    device_type = 'cpu'  # cpu or cuda
    processes = []
    mp.set_start_method("spawn")
    for rank in range(world_size):
        p = mp.Process(target=init_process, args=(rank, world_size, device_type, test_lu))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
