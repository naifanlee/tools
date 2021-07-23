import re

def usort(fnames):
    return sorted(fnames, key=lambda fname:int(re.sub(r'[^0-9]', '', fname)))

def clip(x, xmin, xmax):
    if x < xmin:
        return xmin
    elif x > xmax:
        return xmax
    else:
        return x