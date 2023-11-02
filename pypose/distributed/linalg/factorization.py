import torch
import numpy as np
from datetime import timedelta
import torch.distributed._functional_collectives as funcol
import torch.distributed as dist
from torch.distributed._tensor import Shard,  distribute_tensor

def get_neighbor(i, s):
    if i % 2 ** (s + 1) == 0:
        return i + 2 ** s
    else:
        return i - 2 ** s
def get_null_id(mesh_array, find_ranks): #获取空通信ranks
  result = []
  for idx in mesh_array.flatten():
    if idx not in find_ranks:
      result.append(idx)
  return result

def get_panel_group(mesh_array,current_rank, method = 0): #计算panel的进程
  row_ranks, col_ranks, null_ranks = [], [], []
  row_grp, col_grp, null_grp = None, None, None
  mesh_shape = mesh_array.shape

  if method == 0:
    for row_loop in range(current_rank + 1, mesh_shape[0]):
      row_ranks.append(mesh_array[current_rank, row_loop])
      col_ranks.append(mesh_array[row_loop, current_rank])

  else:
    for row_loop in range(current_rank + 1):
      row_ranks.append(mesh_array[current_rank, row_loop])
      col_ranks.append(mesh_array[row_loop, current_rank])

  null_ranks = get_null_id(mesh_array, row_ranks + col_ranks)

  return row_ranks, col_ranks, null_ranks

def get_panel_broad_rank(mesh_array, rank,method = 0): #获取braodcast ranks

  result,null_result = [], []
  x, y = np.argwhere(mesh_array == rank)[0]
  if method == 0:
    for i in range(x + 1,mesh_array.ndim):
      result.append(mesh_array[i,y])
    if len(result) == 0:
      result = [mesh_array[0,y]]
  else:

    for i in range(y + 1,len(mesh_array)):
      result.append(mesh_array[x,i])

    if len(result) == 0:
        result = [mesh_array[x,0]]

  null_ranks = get_null_id(mesh_array, result)
  return result, null_ranks


def update_row_panel(l, local_data):

  return torch.linalg.solve(l, local_data) #行块更新

def update_col_panel(u, local_data):

  return torch.linalg.solve(u, local_data, left = False) #列块更新

def update_diagblock(local_data, list_l, list_u):
  for i in range(len(list_l)):
    local_data -= list_l[i] @ list_u[i] #对角块更新
  return local_data
def find_index(grp_ranks, input):
  list1 = []
  for item in input[:len(input) - 1]:
    list1.append(np.argwhere(grp_ranks == item)[0])

  return list1

def block_diag(m):  # 需要异构块支持奇数个设备
    """
    https://github.com/pytorch/pytorch/issues/31932#issuecomment-585374009
    """
    if type(m) is list:
        m = torch.stack(m)

    d = m.dim()
    n = m.shape[-3]
    siz0 = m.shape[:-3]
    siz1 = m.shape[-2:]
    m2 = m.unsqueeze(-2)
    eye = attach_dim(torch.eye(n).unsqueeze(-2), d - 3, 1)
    return (m2 * eye).reshape(
        siz0 + torch.Size(torch.tensor(siz1) * n)
    )


def attach_dim(v, n_dim_to_prepend=0, n_dim_to_append=0):
    """
    https://github.com/pytorch/pytorch/issues/31932#issuecomment-585374009
    """

    return v.reshape(
        torch.Size([1] * n_dim_to_prepend)
        + v.shape
        + torch.Size([1] * n_dim_to_append))


class Dfactori:
    r'''
        Performs basic distributed matrix Factorization

        Args: python setup.py develop
            rank(:obj:'int'): Rank for the default process group.
            world_size(:obj:'int'): The world size of the global process group.
            device_type(:obj:'str'): the device type of devicemesh. Default: ``cpu``
        '''

    def __init__(self,  rank, world_size=4, device_type='cpu'):
        self.rank = rank
        self.world_size = world_size
        self.device_type = device_type

    def tslu(self,local_input, max_step):
        s = 0

        P, L, U = torch.linalg.lu(local_input)
        L = torch.matmul(P, L)
        U = U.contiguous()
        list_L = [L]
        list_s = [s]

        while True:

            j = get_neighbor(self.rank, s)
            U_remote = torch.zeros_like(U)
            if j < 0 or j >= self.world_size:
                continue

            if self.rank < j and j < self.world_size:
                req = dist.irecv(U_remote, src=j)
                req.wait()

            elif self.rank > j:
                req = dist.isend(U, dst=j)
                req.wait()
                break

            A_ = torch.cat([U, U_remote])
            s += 1
            P, L, U = torch.linalg.lu(A_)
            L = torch.matmul(P, L)
            U = U.contiguous()
            list_L.append(L)
            list_s.append(s)
            if s == max_step:
                return list_L, U, list_s

        return list_L, U, list_s
    def calu(self):
        return
    def factorization(self, input, device_mesh, method=tslu):
        r'''
        Returns multiplication for tensor.

        Args:
            input (:obj:`Tensor`): input tensor.
            device_mesh(:obj:`DeviceMESH`):  contains the mesh info of ranks.
            method(:obj:`str`): shard method of matrix factorization. Default: ``tslu``

        Return:
            :obj:`Tensor`: the result of multiplication of A, B.

        Note:
            Implementation is based on these papers

            * Gonzalez, Daniel Torres. Application-based fault tolerance for numerical linear
            algebra at large scale. Diss. Université Paris-Nord-Paris XIII, 2021.

        Example:
            >>> A = torch.rand(1000,500)
            >>> L, U = factorization(A, method='tslu')
            >>> L @ U - A
            tensor(-0.0173)
        '''

        if method == 'tslu':
            rank = dist.get_rank()
            max_step = int(np.log2(self.world_size))
            dx = distribute_tensor(input, device_mesh, [Shard(0)])
            U_ = None
            L, U, s = self.tslu(dx.to_local(), max_step)
            L = {self.rank: L}
            if rank == 0:
                print('u_shape', U.shape)
                U_ = U
            l_list = [None for i in range(self.world_size)]
            l_obj = {}
            dist.all_gather_object(l_list, L)
            dist.barrier()
            
            if rank == 0:
                for data in l_list:
                    l_obj.update(data)

                dic_L = {}   #store L in same step
                for i in range(max_step + 1):
                    dic_L.update({i: None})

                for idx in range(self.world_size):
                    data = l_obj[idx]
                    for j in range(len(data)):
                        # print(idx,j,data[j].shape)
                        if dic_L[j] is None:
                            dic_L[j] = [data[j]]
                        else:
                            dic_L[j].append(data[j])

                L_result = None
                for key in dic_L.keys(): #get multi step l multiplication

                    if L_result is None:
                        L_result = block_diag(dic_L[key])
                    else:
                        L_result = torch.matmul(L_result, block_diag(dic_L[key]))
                result = torch.matmul(L_result, U_)
                return result
