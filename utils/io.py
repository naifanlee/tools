import glob
import os
import os.path as osp
from pathlib import Path
import shutil

from .utils import usort

imgtypes = ['jpg', 'bmp', 'png']

def checkpath(p, ok=None):
    p = Path(p)
    if p.is_dir():
        if ok is None:
            print('==> Path: {}, exists.'.format(p.resolve()))
            ok = input('    delete? y/[n]: ')
            ok = 'delete_ok' if ok.strip().lower() == 'y' else 'exist_ok'
        if ok == 'exist_ok':
            pass
        else:  # delte_ok
            shutil.rmtree(p)
            print('    Delete and recreate p: "{}"'.format(p))
            p.mkdir(parents=True)
    else:
        p.mkdir(parents=True)


def walk(p, regex='*'):
    if regex is None:
        regex = '*'
    walks = []
    imgnum = 0
    for dirpath, dirs, fnames in os.walk(p):
        fpaths = glob.glob('{}/{}'.format(dirpath, regex))
        if regex == '*':
            fpaths = [fpath for fpath in fpaths if fpath[-3:] in imgtypes]
            
        if fpaths:
            fpaths = usort(fpaths)
            fpaths = [osp.abspath(fpath) for fpath in fpaths]
            walks.append([osp.dirname(fpaths[0]), fpaths])
            imgnum += len(fpaths)

    print('==> Number: {:<6d}, Path: {}'.format(imgnum, p))
    for dirpath, fpaths in walks:
        print('    Number: {:<6d}, dirpath: {}'.format(len(fpaths), dirpath))

    return walks
