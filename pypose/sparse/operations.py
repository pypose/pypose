
from .sparse_block_tensor import SbkTensor

def abs(sbt):
    assert isinstance(sbt, SbkTensor)
    return sbt.abs()
