import os
import os.path as osp
from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand
import sys


if os.getenv("READTHEDOCS") == "True":
    # to make versioneer working, we need to unshallow this repo
    # because RTD does a checkout with --depth 50
    import subprocess as spr
    rootdir = osp.dirname(__file__)
    spr.call(["git", "-C", rootdir, "fetch", "--unshallow", "origin"])


import versioneer


def readme():
    with open('README.rst') as f:
        return f.read()


class PyTest(TestCommand):
    user_options = [('pytest-args=', 'a', "Arguments to pass to pytest")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = ''

    def run_tests(self):
        import shlex
        # import here, cause outside the eggs aren't loaded
        import pytest
        errno = pytest.main(shlex.split(self.pytest_args))
        sys.exit(errno)


cmdclass = versioneer.get_cmdclass({'test': PyTest})


setup(name='psy-maps',
      version=versioneer.get_version(),
      description='Psyplot plugin for visualization on a map',
      long_description=readme(),
      classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Visualization',
        'Topic :: Scientific/Engineering :: GIS',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: GNU General Public License v2 (GPLv2)',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Operating System :: OS Independent',
      ],
      keywords='visualization netcdf raster cartopy earth-sciences psyplot',
      url='https://github.com/psyplot/psy-maps',
      author='Philipp Sommer',
      author_email='philipp.sommer@unil.ch',
      license="GPLv2",
      packages=find_packages(exclude=['docs', 'tests*', 'examples']),
      install_requires=[
          'psyplot>=1.3.0',
          'psy-simple>=1.3.0',
          'cartopy',
          'scipy',  # required for plotting with cartopy
      ],
      project_urls={
          'Documentation': 'https://psyplot.readthedocs.io/projects/psy-maps',
          'Source': 'https://github.com/psyplot/psy-maps',
          'Tracker': 'https://github.com/psyplot/psy-maps/issues',
      },
      tests_require=['pytest'],
      cmdclass=cmdclass,
      entry_points={'psyplot': ['plugin=psy_maps.plugin',
                                'patches=psy_maps.plugin:patches']},
      zip_safe=False)
