from __future__ import unicode_literals
from distutils import dir_util
from pytest import fixture
import os


@fixture(scope="module")
def datadir(request):
    '''
    Fixture responsible for searching a folder with the same name of test
    module and, if available, moving all contents to a temporary directory so
    tests can use them freely.
    '''
    filename = request.module.__file__
    test_dir, _ = os.path.splitext(filename)
    test_dir = os.path.dirname(test_dir)

    return test_dir