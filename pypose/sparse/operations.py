
from .sbktensor import SbkTensor

def abs(sbt):
    assert isinstance(sbt, SbkTensor)
    return sbt.abs()
