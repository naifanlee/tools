import glob
import sys
from pathlib import Path
from utils.io import annwrite, checkpath
import cv2

from utils import *

source = sys.argv[1]
dst = 'savepath'
checkpath(dst)
txtlist = glob.glob('{}/**/olabels/**/*txt'.format(source), recursive=True)
for i, txtfpath in enumerate(txtlist):
    outtxtfpath = (dst / Path(txtfpath).relative_to(source)).as_posix()
    print(outtxtfpath)
    outtxtpath = Path(outtxtfpath).parent.as_posix()
    print(outtxtpath)
    checkpath(outtxtpath, ok='exist_ok')
    imgfpath = Path(txtfpath).with_suffix('.png').as_posix().replace('label', 'image')
    print(imgfpath)
    imgh, imgw = cv2.imread(imgfpath).shape[:-1]
    print(imgh, imgw)
    with open(txtfpath, 'r') as fr:
        lines = fr.readlines()
        output = []
        for line in lines:
            line = line.strip().split()
            catenm = line[1]
            xlt, ylt, xrb, yrb = map(float, line[2:])
            cateid = catenms_train.index(catenm)

            xlt = clip(xlt, 0, imgw)
            ylt = clip(ylt, 0, imgh)
            xrb = clip(xrb, 0, imgw)
            yrb = clip(yrb, 0, imgh)
            w = xrb - xlt
            h = yrb - ylt
            xc = xlt + w / 2
            yc = ylt + h / 2
            xc /= float(imgw)
            yc /= float(imgh)
            w /= float(imgw)
            h /= float(imgh)

            output.append([cateid, xc, yc, w, h])
        print(i, outtxtfpath)
        annwrite(outtxtfpath, output)
            
        
            

