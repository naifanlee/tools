import re

def usort(fnames):
    if isinstance(fnames[0], list):
        fnames = sorted(fnames, key=lambda k:int(re.sub(r'[^0-9]', '', k[0])))
    elif isinstance(fnames[0], dict):
        fnames = dict(sorted(fnames.items(), key=lambda k:int(re.sub(r'[^0-9]', '', k))))
    else:
        fnames = sorted(fnames, key=lambda fname:int(re.sub(r'[^0-9]', '', fname)))

    return fnames

def clip(x, xmin, xmax):
    if x < xmin:
        return xmin
    elif x > xmax:
        return xmax
    else:
        return x