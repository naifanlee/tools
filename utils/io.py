from pathlib import Path
import shutil



def checkfile(fpath):
    pass

def checkpath(p, ok=None):
    p = Path(p)
    if p.is_dir():
        if ok is None:
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

def uglob(p, regex='*'):
    fpaths = list(Path(p).rglob(regex))
    print('==> Number of files: {}'.format(len(fpaths)))
    return fpaths