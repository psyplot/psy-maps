#!/usr/bin/env python
import sys
import os
from utils import get_ref_dir, get_ref_branch, repo
import subprocess as spr
import shutil
from deploy import deploy
import glob
import os.path as osp

ref_branch = get_ref_branch()
ref_dir = get_ref_dir()

deploy_dir = 'deploy'

work = os.getcwd()

spr.check_call(['git', 'clone', '-b', ref_branch,
                repo.replace('psy-maps', 'psy-maps-references'),
                deploy_dir])

os.chdir(deploy_dir)

spr.check_call('git branch TRAVIS_DEPLOY'.split())

spr.check_call('git checkout TRAVIS_DEPLOY'.split())

os.chdir(work)

for f in glob.glob(osp.join(ref_dir, '*.png')):
    shutil.copyfile(f, osp.join(deploy_dir, osp.basename(f)))

deploy(deploy_dir, ref_branch, '.')


if sys.platform == 'win32':
    spr.check_call('powershell -Command Remove-Item -Recurse -Force'.split() +
                   [deploy_dir])
else:
    shutil.rmtree(deploy_dir)
