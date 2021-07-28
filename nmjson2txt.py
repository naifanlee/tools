import argparse
import glob
import os
import os.path as osp
import sys
from pathlib import Path

import cv2

from utils import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgpath', type=str, required=True)
    parser.add_argument('--dst', type=str, default=None)
    parser.add_argument('--stream', action='store_true')
    args = parser.parse_args()
    
    args.stream = int(args.stream)
    args.dst_w, args.dst_h = 640, 384
    if args.dst:
        checkpath(args.dst)

    return args

if __name__ == '__main__':
    args = parse_args()
    import time
    stime = time.time()

    cv2.namedWindow('viewimg', cv2.WINDOW_NORMAL)

    for dirpath, fpaths in walk(args.imgpath, regex='oimages_anno/*.bmp'):
        annjson_fpath = glob.glob(osp.join(dirpath, '../*nm*.json'))[0]
        print(annjson_fpath)
       
        anns_dt = parse_nmjson(annjson_fpath)

        # check dst path
        rel_fpath = Path(dirpath).relative_to(args.imgpath)
        dst = Path(args.dst)
        simg_path = dst / 'images' / rel_fpath
        oimg_path = dst / 'oimages' / rel_fpath
        sann_path = dst / 'slabels' / rel_fpath
        oann_path = dst / 'olabels' / rel_fpath
        sann_norm_path = dst / 'slabels_norm' / rel_fpath
        checkpath(simg_path)
        checkpath(oimg_path)
        checkpath(sann_path)
        checkpath(oann_path)
        checkpath(sann_norm_path)

        for imgfpath in fpaths:
            fname = Path(imgfpath).with_suffix('.txt').name
            sann_fpath = sann_path / fname
            oann_fpath = oann_path / fname
            sann_norm_fpath = sann_norm_path / fname

            img = cv2.imread(imgfpath)
            oanns = anns_dt.get(fname[:-4], [])
            draw_bboxes(img, oanns)
            cv2.imshow('viewimg', img)
            key = cv2.waitKey(args.stream)
            if key == ord('q'):
                sys.exit(0)
            
            # write_anns(sann_fpath, sanns)
            # write_anns(oann_fpath, oanns)
            # write_anns(sann_norm_path, sanns_norm)
            # cv2.imwrite(simg_fpath, img)
            # cv2.imwrite(oimg_fpath, img)                

    cv2.destroyAllWindows()        

