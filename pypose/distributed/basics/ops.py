import torch
import torch.distributed as dist
from torch.distributed._tensor import DeviceMesh, Shard, Replicate, distribute_tensor, zeros


class Dops:
    r'''
    Performs basic distributed tensor operator

    Args:
        world_size(:obj:'Int'): The world size of the global process group.
        device_type(:obj:'Int'): the device type of devicemesh. Default: ``cpu``
    '''

    def __init__(self, world_size=4, device_type='cpu'):
        self.world_size = world_size
        self.device_type = device_type

    def matmulAB(self, A, B, mesh_shape, method='row'):
        r'''
            Returns multiplication for tensor.

            Args:
                A (:obj:`Tensor`): left input tensor.
                B (:obj:`Tensor`): right input tensor.
                mesh_shape(:obj:`tuple`): shape of device_mesh.
                method(:obj:`str`): shard method of multiplication. Default: ``row``

            Return:
                :obj:`Tensor`: the result of multiplication of A, B.

            Note:
                Implementation is based on these papers

                * Cheng, Shenggan, et al. "ATP: Adaptive Tensor Parallelism for Foundation
                  Models." arXiv preprint arXiv:2301.08658 (2023).

            Example:
                >>> A = torch.rand(3000,5000)
                >>> B = torch.rand(5000,2000)
                >>> matmulAB(A,B).shape
                tensor([3000,2000])
            '''
        assert A.shape[1] == B.shape[0], "The shape of the matrix of A and B is illegal"
        assert mesh_shape[0] * mesh_shape[1] == self.world_size, 'mesh_shape[0] * mesh_shape[1]' \
                                                                 ' must euqal to world_size'
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size).reshape(mesh_shape))
        if method == 'row':
            dA = distribute_tensor(A, mesh, [Shard(1), Replicate()])
            dB = distribute_tensor(B, mesh, [Shard(0), Shard(1)])
            out_shape = [A.shape[0], int(B.shape[1] / 2)]
            output = [torch.zeros(out_shape, dtype=torch.float32) for _ in range(mesh_shape[1])]
            result = torch.matmul(dA.to_local(), dB.to_local()).to('cpu')
            dist.all_reduce(result, op=dist.ReduceOp.SUM, async_op=False,
                            group=mesh.get_dim_groups(0))
            dist.all_gather(output, result, group=mesh.get_dim_groups(1))
            dist.barrier()
            return torch.cat(output, dim=1)

    def matmulABT(self, A, B, mesh_shape, method='row'):
        pass

    def matmulATB(self, A, B, mesh_shape, method='row'):
        pass

    def summa(self, A, B, mesh_shape):
        pass
