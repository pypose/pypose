
import unittest

from tests.unit_tests.test_ba_layer import *
from tests.unit_tests.test_functorch import *
from tests.unit_tests.test_reprojection_error import *
from tests.unit_tests.test_sparse_block_matrix import *

if __name__ == '__main__':
    print('Run all the unit tests. ')
    unittest.main()