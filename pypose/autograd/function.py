import torch
from functools import wraps

from .. import _format_sparse_backend_error, _load_optional_backend_attr


def _missing_tracking_tensor():
    class TrackingTensor(torch.Tensor):
        r"""Placeholder for the optional sparse backend tracking tensor."""

        @staticmethod
        def __new__(cls, data, *args, **kwargs):
            raise ImportError(
                _format_sparse_backend_error("pypose.autograd.function.TrackingTensor")
            )

    return TrackingTensor


def _missing_map_transform():
    def map_transform(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            raise ImportError(
                _format_sparse_backend_error("pypose.autograd.function.map_transform")
            )

        return wrapped

    return map_transform


def __getattr__(name):
    if name == "TrackingTensor":
        value, _ = _load_optional_backend_attr("bae.autograd.function", name)
        value = _missing_tracking_tensor() if value is None else value
        globals()[name] = value
        return value
    if name == "map_transform":
        value, _ = _load_optional_backend_attr("bae.autograd.function", name)
        value = _missing_map_transform() if value is None else value
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | {"TrackingTensor", "map_transform"})


__all__ = ["TrackingTensor", "map_transform"]
