from packaging import version

from ._version import __version__

from .lietensor import SO3_Log, SE3_Log, RxSO3_Log, Sim3_Log
from .lietensor import so3_Exp, se3_Exp, rxso3_Exp, sim3_Exp
from .lietensor import SO3_Act, SE3_Act, RxSO3_Act, Sim3_Act
from .lietensor import SO3_Mul, SE3_Mul, RxSO3_Mul, Sim3_Mul
from .lietensor import SO3_Inv, SE3_Inv, RxSO3_Inv, Sim3_Inv
from .lietensor import SO3_Act4, SE3_Act4, RxSO3_Act4, Sim3_Act4
from .lietensor import SO3_AdjXa, SE3_AdjXa, RxSO3_AdjXa, Sim3_AdjXa
from .lietensor import SO3_AdjTXa, SE3_AdjTXa, RxSO3_AdjTXa, Sim3_AdjTXa
from .lietensor import so3_Jl_inv, se3_Jl_inv, rxso3_Jl_inv, sim3_Jl_inv
from .lietensor import broadcast_inputs

from .lietensor import LieTensor, Parameter, SO3, so3, SE3, se3, Sim3, sim3, RxSO3, rxso3
from .lietensor import randn_like, randn_SE3, randn_SO3, randn_so3, randn_se3
from .lietensor import randn_Sim3, randn_sim3, randn_RxSO3, randn_rxso3
from .lietensor import identity_like, identity_SO3, identity_so3, identity_SE3, identity_se3
from .lietensor import identity_Sim3, identity_sim3, identity_RxSO3, identity_rxso3
from .lietensor import add, add_, mul, Exp, Log, Inv, Mul, Retr, Act, Adj, AdjT, Jinvp, Jr
from .lietensor import SO3_type, so3_type, SE3_type, se3_type, SE3Type
from .lietensor import Sim3_type, sim3_type, RxSO3_type, rxso3_type
from .lietensor import tensor, translation, rotation, scale, matrix, euler
from .lietensor import mat2SO3, mat2SE3, mat2Sim3, mat2RxSO3, from_matrix, matrix, euler2SO3, vec2skew
from .function import *
from .basics import *
from . import module
from . import optim



min_torch = '2.0'
assert version.parse(min_torch) <= version.parse(torch.__version__), \
    f'PyTorch=={torch.__version__} is used but incompatible. ' \
    f'Please install torch>={min_torch}.'
