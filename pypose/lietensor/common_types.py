from typing import Union, Tuple, Optional
import collections
from itertools import repeat
import torch

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

_size_any_t = Union[float, Tuple[float, ...], torch.Tensor]
