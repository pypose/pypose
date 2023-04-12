
from .sparse_block_tensor import SparseBlockTensor

def abs(sbt):
    assert isinstance(sbt, SparseBlockTensor)
    return sbt.abs()