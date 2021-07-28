
import argparse
import os
import sys
from pathlib import Path

import cv2

from utils import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgpath', nargs='+', type=str, required=True)
    parser.add_argument('--annopath', type=str, default=None)
    parser.add_argument('--dst', default=None)
    parser.add_argument('--regex', type=str, default='*.bmp')
    parser.add_argument('--stream', action='store_true')
    args = parser.parse_args()
    args.stream = int(args.stream)
    if not args.annopath:
        args.annopath = args.imgpath
    if args.dst:
        checkpath(args.dst)
    print(args)

    return args

if __name__ == '__main__':
    args = parse_args()


    imgpath = args.imgpath[0]
    cv2.namedWindow('viewimg', cv2.WINDOW_NORMAL)    
    for dirpath, fpaths in walk(imgpath, regex=args.regex):
        idx = 0
        while 1:
            imgfpath = Path(fpaths[idx])
            print(imgfpath)

            rel_fpath = imgfpath.relative_to(imgpath)
            fname = imgfpath.name
            if args.annopath:
                rel_annofpath = rel_fpath.with_suffix('.txt').as_posix().replace('/images/',  '/labels/')
                annofpath = (Path(args.annopath) / rel_annofpath).as_posix()

            img = cv2.imread(str(imgfpath))
            draw_bboxes(img, annofpath, show_conf=True)
            cv2.imshow('viewimg', img)

            key = cv2.waitKey(args.stream)
            if args.dst:
                dst = args.dst / rel_fpath.parent
                checkpath(dst, ok='exist_ok')
                cv2.imwrite((dst / fname).as_posix(), img)
            if key == ord('q'):
                sys.exit(0)
            elif key == ord('b'):
                idx -= 1
            else:
                idx += 1
            
            if idx + 1 > len(fpaths):
                break

        if args.dst:
            print('\nResults saved to {}\n'.format(Path(args.dst).resolve()))
    
    cv2.destroyAllWindows()        


