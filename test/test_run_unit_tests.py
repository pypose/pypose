
import unittest

from unit_tests.test_ba_layer import *
from unit_tests.test_cu_sparse_block_tensor import *
from unit_tests.test_functorch import *
from unit_tests.test_linear_solver_cu_sparse import *
from unit_tests.test_reprojection_error import *
from unit_tests.test_sparse_block_tensor import *

if __name__ == '__main__':
    print('Run all the unit tests. ')
    unittest.main()