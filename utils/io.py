import glob
import os
import os.path as osp
from pathlib import Path
import shutil

from .utils import usort

imgtypes = ['.jpg', '.bmp', '.png']

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


def walk(p, regex='**/*'):
    regex = '**/*' if regex == '*' else regex
    fpaths = glob.glob('{}/{}'.format(p, regex), recursive=True)
    fpaths = [fpath for fpath in fpaths if fpath[-4:] in imgtypes]
    print('==> Number: {:<6d}, Path: {}'.format(len(fpaths), '{}/{}'.format(osp.abspath(p), regex)))
    assert(len(fpaths)), 'No images in p: {}'.format(p)
    if fpaths:
        walks = {}
        for fpath in fpaths:
            dirpath = Path(fpath).parent.as_posix()
            if dirpath not in walks:
                walks[dirpath] = [fpath]
            else:
                walks[dirpath].append(fpath)

        for dirpath, fpaths in walks.items():
            try:
                walks[dirpath] = usort(fpaths)
            except:
                walks[dirpath] = fpaths
        try:
            walks = usort(walks).items()
        except:
            walks = walks.items()

    for dirpath, fpaths in walks:
        print('    Number: {:<6d}, dirpath: {}'.format(len(fpaths), dirpath))

    return walks

def annwrite(annfpath, anns):
    if osp.exists(annfpath):
        os.system('rm -rf {}'.format(annfpath))
    with open(annfpath, 'a') as fa:
        for ann in anns:
            fa.write(' '.join(list(map(lambda x: str(x), ann))) + '\n')