"""
.. module:: fileutil
   :synopsis: Unit tests for fileutil module
"""

import os
import os.path as op
import shutil

import nutsml.fileutil as fu
import pytest


@pytest.fixture()
def init_test_folders(request):
    """Remove folder 'foo' and sub-folders at setup and teardown."""
    path = op.join("data", "foo")

    def cleanup():
        if os.path.exists(path):
            shutil.rmtree(path)

    cleanup()
    request.addfinalizer(cleanup)
    return path


def test_create_filename():
    assert len(fu.create_filename()) > 0
    assert fu.create_filename('prefix', '').startswith('prefix')
    assert fu.create_filename('', 'ext').endswith('.ext')


def test_create_filename_is_unique():
    # Create set of 100 file names and verify that they are unique.
    nameset = {fu.create_filename() for _ in xrange(100)}
    assert len(nameset) == 100


def test_create_temp_filepath():
    assert fu.create_temp_filepath().startswith(fu.TEMP_FOLDER)
    assert fu.create_temp_filepath(relative=False).startswith(os.getcwd())
    assert fu.create_temp_filepath('', 'ext').endswith('.ext')
    assert os.path.exists(fu.TEMP_FOLDER), "temp folder should exist"
    fu.delete_folders(fu.TEMP_FOLDER)  # cleaning up.


def test_delete_file():
    path = 'data/' + fu.create_filename(ext='txt')
    fu.delete_file(path)  # file does not exist. Should be fine.
    with open(path, 'w') as f:
        f.write('foo')
    assert os.path.exists(path)
    fu.delete_file(path)
    assert not os.path.exists(path), "files should be deleted"


def test_create_folders(init_test_folders):
    path = init_test_folders
    fu.create_folders(path)  # make new folder.
    assert os.path.exists(path), "foo should exist"
    fu.create_folders(path)  # make foo again.
    assert os.path.exists(path), "foo should still exist"
    fu.create_folders(op.join(path, "bar"))
    assert os.path.exists(path), "foo/bar should exist"


def test_delete_folders(init_test_folders):
    path = init_test_folders
    fu.delete_folders(path)  # delete non-existing folder is fine.
    os.makedirs(path)
    fu.delete_folders(path)  # delete existing folder.
    assert not os.path.exists(path), "foo should not exist"
    os.makedirs(op.join(path, "bar"))
    fu.delete_folders(path)
    assert not os.path.exists(path), "foo should not exist"


def test_delete_temp_data():
    fu.create_folders(fu.TEMP_FOLDER)
    fu.delete_temp_data()
    assert not os.path.exists(fu.TEMP_FOLDER), "temp folder should not exist"


def test_clear_folder(init_test_folders):
    path = init_test_folders
    bardir, bazfile = op.join(path, "bar"), op.join(path, "baz.txt")
    os.makedirs(bardir)
    open(bazfile, "w").close()
    fu.clear_folder(path)
    assert os.path.exists(path), "foo folder should exist"
    assert not os.path.exists(bardir), "bar folder should not exist"
    assert not os.path.isfile(bazfile), "baz file should not exist"
