from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version as get_version
from importlib import import_module
from packaging import version
from packaging.specifiers import SpecifierSet

from ._version import __version__

_BAE_VERSION_SPEC = SpecifierSet(">=0.2,<0.3")
_BAE_KNOWN_COMPATIBLE_VERSION = "0.2"
_BAE_INSTALL_COMMAND = "pip install git+https://github.com/sair-lab/bae.git" + f"@{_BAE_KNOWN_COMPATIBLE_VERSION}"


def _format_sparse_backend_error(feature):
    return (
        f"PyPose requires bae{_BAE_VERSION_SPEC} when "
        f"the optional backend is used. \n"
        f"Recommend running (known-compatible):\n "
        f"   {_BAE_INSTALL_COMMAND}"
    )


def _ensure_sparse_backend_version():
    try:
        installed_version = get_version("bae")
    except PackageNotFoundError:
        return

    if version.parse(installed_version) not in _BAE_VERSION_SPEC:
        raise ImportError(
            f"PyPose requires bae{_BAE_VERSION_SPEC} when "
            f"the optional backend is installed, but found "
            f"bae=={installed_version}. \n Recommend running (known-compatible):\n "
            f"   {_BAE_INSTALL_COMMAND}"
        )


@lru_cache(maxsize=None)
def _load_optional_backend_attr(module_name, attr_name):
    try:
        module = import_module(module_name)
    except ImportError as exc:
        return None, exc
    return getattr(module, attr_name), None


def _require_backend_attr(module_name, attr_name, feature):
    attr, exc = _load_optional_backend_attr(module_name, attr_name)
    if attr is None:
        raise ImportError(_format_sparse_backend_error(feature)) from exc
    return attr


_ensure_sparse_backend_version()

from .lietensor import LieTensor, Parameter, SO3, so3, SE3, se3, Sim3, sim3, RxSO3, rxso3
from .lietensor import randn_like, randn_SE3, randn_SO3, randn_so3, randn_se3
from .lietensor import randn_Sim3, randn_sim3, randn_RxSO3, randn_rxso3
from .lietensor import identity_like, identity_SO3, identity_so3, identity_SE3, identity_se3
from .lietensor import identity_Sim3, identity_sim3, identity_RxSO3, identity_rxso3
from .lietensor import add, add_, mul, Exp, Log, Inv, Mul, Retr, Act, Adj, AdjT, Jinvp, Jr
from .lietensor import SO3_type, so3_type, SE3_type, se3_type
from .lietensor import Sim3_type, sim3_type, RxSO3_type, rxso3_type
from .lietensor import tensor, translation, rotation, scale, matrix, euler, quat2unit
from .lietensor import mat2SO3, mat2SE3, mat2Sim3, mat2RxSO3, from_matrix, matrix, euler2SO3, vec2skew
from .lietensor.lietensor import retain_ltype
from . import func
from .function import *
from .basics import *
from .sparse import *
from . import autograd
from . import module
from . import optim
from . import testing
from . import metric
from .module.loss import geodesic_loss

min_torch = '2.0'
assert version.parse(min_torch) <= version.parse(torch.__version__), \
    f'PyTorch=={torch.__version__} is used but incompatible. ' \
    f'Please install torch>={min_torch}.'
