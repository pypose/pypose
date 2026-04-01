import torch
from functools import wraps

from .. import _format_sparse_backend_error, _load_optional_backend_attr


def _missing_track():
    class Track(torch.Tensor):
        r"""Placeholder for the optional sparse backend tracking tensor."""

        @staticmethod
        def __new__(cls, data, *args, **kwargs):
            raise ImportError(
                _format_sparse_backend_error("pypose.autograd.function.Track")
            )

    return Track


def _load_track_aliases():
    value, _ = _load_optional_backend_attr("bae.autograd.function", "TrackingTensor")
    value = _missing_track() if value is None else value
    globals()["Track"] = value
    globals()["TrackingTensor"] = value
    return value


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
    if name in {"Track", "TrackingTensor"}:
        return _load_track_aliases()
    if name == "map_transform":
        value, _ = _load_optional_backend_attr("bae.autograd.function", name)
        value = _missing_map_transform() if value is None else value
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | {"Track", "TrackingTensor", "map_transform"})


__all__ = ["Track", "TrackingTensor", "map_transform"]
