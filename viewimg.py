
import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

from utils import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgpaths', nargs='+', type=str, required=True)
    parser.add_argument('--annpaths', type=str, default=[])
    parser.add_argument('--dst', default=None)
    parser.add_argument('--regex', type=str, default='*')
    parser.add_argument('--stream', action='store_true')
    args = parser.parse_args()
    args.stream = int(args.stream)
    if args.dst:
        checkpath(args.dst)
    print(args)

    return args

if __name__ == '__main__':
    args = parse_args()


    imgpath = args.imgpaths[0]
    cv2.namedWindow('viewimg', cv2.WINDOW_NORMAL)    
    for dirpath, fpaths in walk(imgpath, regex=args.regex):
        idx = 0
        while 1:
            imgfpath = Path(fpaths[idx])
            print('==> {}'.format(imgfpath))

            img1 = cv2.imread(str(imgfpath))
            rel_imgfpath = imgfpath.relative_to(imgpath)
            if len(args.annpaths) >= 1:
                rel_annofpath = rel_imgfpath.with_suffix('.txt').as_posix().replace('/images/',  '/labels/')
                annofpath = (Path(args.annpaths[0]) / rel_annofpath).as_posix()
                anns = parse_args(annofpath)
                draw_bboxes(img1, anns, show_conf=True)

            if len(args.imgpaths) >= 2:
                img2 = cv2.imread(str(args.imgpaths[1] / rel_imgfpath))
                if len(args.annpaths) >= 2:
                    annofpath = (Path(args.annpaths[1]) / rel_annofpath).as_posix()
                    anns = parse_args(annofpath)
                    draw_bboxes(img2, anns, show_conf=True)
            if len(args.imgpaths) >= 3:
                img3 = cv2.imread(str(args.imgpaths[2] / rel_imgfpath))
                if len(args.annpaths) >= 3:
                    annofpath = (Path(args.annpaths[2]) / rel_annofpath).as_posix()
                    anns = parse_args(annofpath)
                    draw_bboxes(img3, anns, show_conf=True)
            if len(args.imgpaths) >= 4:
                img4 = cv2.imread(str(args.imgpaths[3] / rel_imgfpath))
                if len(args.annpaths) >= 4:
                    annofpath = (Path(args.annpaths[3]) / rel_annofpath).as_posix()
                    anns = parse_args(annofpath)
                    draw_bboxes(img3, anns, show_conf=True)
    

            # imgshow
            if len(args.imgpaths) == 1:
                img_show = img1
            elif len(args.imgpaths) == 2:
                img_show = np.vstack([img1, img2])
            elif len(args.imgpaths) == 3:
                img_show = np.vstack([np.hstack([img1, img2]), np.hstack([img3, np.zeros_like(img3)])])
            elif len(args.imgpaths) == 4:
                img_show = np.vstack([np.hstack([img1, img2]), np.hstack([img3, img4])])
            else:
                assert(False)
            cv2.imshow('viewimg', img_show)
            if args.dst:
                dst = args.dst / rel_imgfpath.parent
                checkpath(dst, ok='exist_ok')
                cv2.imwrite((dst / rel_imgfpath.name).as_posix(), img1)

            key = cv2.waitKey(args.stream)
            if key == ord('q'):
                sys.exit(0)
            elif key == ord('j'):
                idx += 100
            elif key == ord('k'):
                idx -= 100
            if key == ord('b'):
                idx -= 1
            else:
                idx += 1
            
            if idx + 1 > len(fpaths):
                break

        if args.dst:
            print('\nResults saved to {}\n'.format(Path(args.dst).resolve()))
    
    cv2.destroyAllWindows()        


