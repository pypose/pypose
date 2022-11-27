import torch

class SparseBlockTensor(torch.sparse_coo_tensor):
    def __init__(self, *data):
        pass
    
    @staticmethod
    def __new__(cls, *data):
        sbt = data[0] if isinstance(data[0], torch.sparse_coo_tensor) else torch.sparse_coo_tensor(*data)
        return torch.sparse_coo_tensor.as_subclass( sbt, SparseBlockTensor)