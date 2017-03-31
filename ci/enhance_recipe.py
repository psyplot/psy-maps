"""Add the missing informations to the conda recipe
"""
import os
import os.path as osp

with open(osp.join('recipe', 'meta.template')) as f:
    s = f.read()

with open(osp.join('psy_maps', 'version.py')) as f:
    exec(f.read())

s = s.replace('VERSION', __version__)
s = s.replace('PWD', os.getcwd())

with open(osp.join('recipe', 'meta.yaml'), 'w') as f:
    f.write(s)
