from abc import ABC
from torch import nn


class Prepare(ABC):
    r'''
    A class to create a class instance for multiple times, which useful for a class
    that cannot be reset or reinstated, e.g., classes in `torch.optim.lr_scheduler`.

    Args:
        class_name: a class to prepare
        args: argument list
        kwargs: keyword argumet list
    '''
    def __init__(self, class_name, *args, **kwargs):
        self.class_name = class_name
        self.args = args
        self.kwargs = kwargs

    def __call__(self):
        return self.class_name(*self.args, **self.kwargs)
