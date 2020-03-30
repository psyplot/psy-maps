#!/usr/bin/env python
import os
import os.path as osp
from utils import get_ref_dir, get_ref_branch

import subprocess as spr

work = osp.abspath(os.getcwd())

maybe_upload = (
    (os.getenv('TRAVIS') and os.getenv('TRAVIS_PULL_REQUEST') == 'false' and
     os.getenv('TRAVIS_REPO_SLUG') == 'psyplot/psy-maps') or
    (os.getenv('APPVEYOR') and
     os.getenv('APPVEYOR_REPO_NAME') == 'psyplot/psy-maps'))

if maybe_upload:
    os.chdir(get_ref_dir())
    spr.check_call('git add -N .'.split())
    if spr.call('git diff --exit-code'.split()):
        print("------------------------------")
        print("ATTENTION! REFERENCES CHANGED!")
        print("------------------------------")
        print("Enabled the deploy to psy-maps-references")
        open(osp.join(work, 'deploy_references'), 'w').close()
    else:
        print("No changes to the reference figures on this push -- No deploy.")
