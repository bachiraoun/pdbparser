#!/Library/Frameworks/Python.framework/Versions/7.0/Resources/Python.app/Contents/MacOS/Python
from __future__ import print_function
import unittest
import sys, os

# create test suite and runner
__TESTS_SUITE__ = unittest.TestSuite()
__RUNNER__ = unittest.TextTestRunner(verbosity = 2)

# get all tests
path = os.path.join( os.getcwd().split("pdbparser")[0], "pdbparser")
for all_test_suite in unittest.defaultTestLoader.discover('', pattern='Test*.py', top_level_dir = path):
        for test_suite in all_test_suite:
            __TESTS_SUITE__.addTests(test_suite)


def run_all_tests():
    __RUNNER__.run(__TESTS_SUITE__)


if __name__ == "__main__":
    import sys, os
    path = os.path.join( os.getcwd().split("pdbparser")[0], "pdbparser")
    sys.path.insert(0,path)

    import pdbparser
    run_all_tests()
