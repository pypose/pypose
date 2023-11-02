import os
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import pypose as pp
import time


def test_mm_ab(rank, world_size, device_type):
    dops = pp.Dops(rank, world_size, device_type)
    A = torch.rand(1000, 200)
    B = torch.rand(200, 100)
    mesh_shape = (2, 2)
    st_dist = time.perf_counter()
    dresult = dops.matmulAB(A, B, mesh_shape)

    if rank == 0:
        ed_dist = time.perf_counter()
        st = time.perf_counter()
        result = torch.matmul(A, B)
        ed = time.perf_counter()
        time_dist = ed_dist - st_dist
        time_torch = ed - st
        print('Time cost:', time_dist - time_torch)
        print(torch.sum(dresult - result))


def test_mm_abt():
    pass


def test_mm_atb():
    pass


def test_summa():
    pass


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
        p = mp.Process(target=init_process, args=(rank, world_size, device_type, test_mm_ab))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
