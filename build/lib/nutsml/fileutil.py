"""
.. module:: fileutil
   :synopsis: File system utilities.
"""

import glob
import os
import os.path as op
import shutil
import uuid

TEMP_FOLDER = 'temp'


def create_filename(prefix='', ext=''):
    """
    Create a unique filename.

    :param str prefix: Prefix to add to filename.
    :param str ext: Extension to append to filename, e.g. 'jpg'
    :return: Unique filename.
    :rtype: str
    """
    suffix = '.' + ext if ext else ''
    return prefix + str(uuid.uuid4()) + suffix


def create_temp_filepath(prefix='', ext='', relative=True):
    """
    Create a temporary folder under :py:data:`TEMP_FOLDER`.

    If the folder already exists do nothing. Return relative (default) or
    absolute path to a temp file with a unique name.

    See related function :func:`.create_filename`.

    :param str prefix: Prefix to add to filename.
    :param str ext: Extension to append to filename, e.g. 'jpg'
    :param bool relative: True: return relative path, otherwise absolute path.
    :return: Path to file with unique name in temp folder.
    :rtype: str
    """
    create_folders(TEMP_FOLDER)
    rel_path = op.join(TEMP_FOLDER, create_filename(prefix, ext))
    return rel_path if relative else op.abspath(rel_path)


def create_folders(path, mode=0777):
    """
    Create folder(s). Don't fail if already existing.

    See related functions :func:`.delete_folders` and :func:`.clear_folder`.

    :param str path: Path of folders to create, e.g. 'foo/bar'
    :param int mode: File creation mode, e.g. 0777
    """
    if not os.path.exists(path):
        os.makedirs(path, mode)


def delete_file(path):
    """
    Remove file at given path. Don't fail if non-existing.

    :param str path: Path to file to delete, e.g. 'foo/bar/file.txt'
    """
    if os.path.exists(path):
        os.remove(path)


def delete_folders(path):
    """
    Remove folder and sub-folders. Don't fail if non-existing or not empty.

    :param str path: Path of folders to delete, e.g. 'foo/bar'
    """
    if os.path.exists(path):
        shutil.rmtree(path)


def delete_temp_data():
    """
    Remove :py:data:`TEMP_FOLDER` and all its contents.
    """
    delete_folders(TEMP_FOLDER)


def clear_folder(path):
    """
    Remove all content (files and folders) within the specified folder.

    :param str path: Path of folder to clear.
    """
    for sub_path in glob.glob(op.join(path, "*")):
        if os.path.isfile(sub_path):
            os.remove(sub_path)
        else:
            shutil.rmtree(sub_path)
