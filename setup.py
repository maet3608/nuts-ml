import os
import sys
import glob
import shutil
import nutsml

from setuptools import setup, find_packages, Command
from setuptools.command.test import test as TestCommand


class CleanCommand(Command):
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        for folder in ['build', 'dist']:
            if os.path.exists(folder):
                shutil.rmtree(folder)
        for egg_file in glob.glob('*egg-info'):
            shutil.rmtree(egg_file)


class PyTest(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        import pytest
        errcode = pytest.main(self.test_args)
        sys.exit(errcode)


def load_readme():
    with open('README.rst') as f:
        return f.read()


setup(
    name=nutsml.__name__,
    version=nutsml.__version__,
    url='https://maet3608.github.io/nuts-ml',
    download_url='https://github.com/maet3608/nuts-ml',
    license='Apache Software License ('
            'http://www.apache.org/licenses/LICENSE-2.0)',
    author='Stefan Maetschke',
    author_email='stefan.maetschke@gmail.com',
    description='Flow-based data pre-processing for Machine Learning',
    long_description=load_readme(),
    long_description_content_type='text/x-rst',
    install_requires=[
        'nutsflow >= 1.2.3',
        'openpyxl >= 3.0.5',
        'pandas > 0.21.0',
        'six >= 1.10.0',
        'scipy >= 0.17.0',
        'pillow >= 3.0.0',
        'scikit-image >= 0.12.3',
    ],
    tests_require=['pytest >= 3.0.3'],
    platforms='any',
    packages=find_packages(exclude=['setup']),
    include_package_data=True,
    cmdclass={
        'test': PyTest,
        'clean': CleanCommand,
    },
    keywords=['machine learning', 'deep learning', 'image processing'],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Natural Language :: English',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
