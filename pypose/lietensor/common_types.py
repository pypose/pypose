from typing import TypeVar, Union, Tuple, Optional
import collections
from itertools import repeat

def convert_sig_se(sigma):
    if not isinstance(sigma, collections.abc.Iterable):
        sigma = _quadruple(sigma)
    elif len(sigma)==2:
        rotation_sigma = _single(sigma[-1])
        transation_sigma = _triple(sigma[0])
        sigma = transation_sigma+rotation_sigma
    else:
        assert len(sigma)==4
    return sigma

def convert_sig_sim(sigma):
    if not isinstance(sigma, collections.abc.Iterable):
        sigma = _penta(sigma)
    elif len(sigma)==3:
        rotation_sigma = _single(sigma[-2])
        scale_sigma = _single(sigma[-1])
        transation_sigma = _triple(sigma[0])
        sigma = transation_sigma+rotation_sigma+scale_sigma
    else:
        assert len(sigma)==5
    return sigma

def convert_sig_rxs(sigma):
    if not isinstance(sigma, collections.abc.Iterable):
        sigma = _pair(sigma)
    else:
        assert len(sigma)==2
    return sigma

# import from torch
def _ntuple(n, name="parse"):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple(repeat(x, n))

    parse.__name__ = name
    return parse

_single = _ntuple(1, "_single")
_pair = _ntuple(2, "_pair")
_triple = _ntuple(3, "_triple")
_quadruple = _ntuple(4, "_quadruple")
_penta = _ntuple(5, "_penta")

# Create some useful type aliases
T = TypeVar('T')
_scalar_or_tuple_any_t = Union[T, Tuple[T, ...]] # for all lie type
_scalar_or_tuple_24_t = Union[T, Tuple[T, T], Tuple[T, T, T, T]] # for SE3 se3
_scalar_or_tuple_2_t = Union[T, Tuple[T, T]] # for RxSO3, rxso3
_scalar_or_tuple_35_t = Union[T, Tuple[T, T, T], Tuple[T, T, T, T, T]] # for Sim3, sim3

# For arguments which represent sigma parameters (eg, kernel size, padding)
_size_any_t = _scalar_or_tuple_any_t[float]
_size_2_t = _scalar_or_tuple_2_t[float]
_size_35_t = _scalar_or_tuple_35_t[float]
_size_24_t = _scalar_or_tuple_24_t[float]
