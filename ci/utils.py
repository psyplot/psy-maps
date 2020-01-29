import os
import os.path as osp
import six

import subprocess as spr

if six.PY2:
    import imp
    gt = imp.load_source('get_ref_dir', osp.join('tests', 'get_ref_dir.py'))
else:
    import importlib.util as iutil
    spec = iutil.spec_from_file_location('get_ref_dir',
                                         osp.join('tests', 'get_ref_dir.py'))
    gt = iutil.module_from_spec(spec)
    spec.loader.exec_module(gt)

gt.MPL_VERSION = os.getenv('MPL_VERSION')
gt.PY_VERSION = os.getenv('PYTHON_VERSION')


get_ref_dir = gt.get_ref_dir
get_ref_branch = gt.get_ref_branch


p = spr.Popen(
    ('git -C %s config remote.origin.url' % osp.dirname(__file__)).split(),
    stdout=spr.PIPE)
p.wait()
repo = p.stdout.read().decode('utf-8').splitlines()[0]
