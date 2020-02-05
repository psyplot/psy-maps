"""Get the right branch for the psy-simple reference figures"""
import sys
import os
import os.path as osp
import argparse


MPL_VERSION = None
PY_VERSION = None

def get_versions():
    platform = sys.platform
    if MPL_VERSION:
        mpl_version = MPL_VERSION
    else:
        import matplotlib as mpl
        mpl_version = mpl.__version__.rsplit('.', 1)[0]
    if PY_VERSION:
        py_version = PY_VERSION
    else:
        py_version = '.'.join(map(str, sys.version_info[:2]))
    return platform, py_version, mpl_version

def get_ref_dir():
    platform, py_version, mpl_version = get_versions()
    return os.getenv("PSYPLOT_REFERENCES") or osp.join(
        osp.dirname(__file__), 'reference_figures', platform,
        'py' + py_version, 'mpl' + mpl_version)


def get_ref_branch():
    os, py_version, mpl_version = get_versions()
    return '_'.join([platform, 'py' + py_version, 'mpl' + mpl_version])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-b', '--branch', action='store_true',
                        help="Display the reference branch")
    parser.add_argument('-mpl', '--matplotlib',
                        help="The matplotlib version to use")

    parser.add_argument('-py', '--python', help="The python version to use")

    args = parser.parse_args()

    MPL_VERSION = args.matplotlib
    PY_VERSION = args.python

    if args.branch:
        print(get_ref_branch())
    else:
        print(get_ref_dir())
