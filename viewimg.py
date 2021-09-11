
import argparse
from genericpath import exists, isfile
import os
import sys
from pathlib import Path

import cv2
import numpy as np

from utils import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgpaths', nargs='+', type=str, required=True)
    parser.add_argument('--annpaths', nargs='+', type=str, default=None)
    parser.add_argument('--dst', default=None)
    parser.add_argument('--regex', type=str, default='*')
    parser.add_argument('--stream', action='store_true')
    parser.add_argument('--filter', default=None)
    parser.add_argument('--classes', nargs='+', type=str, default=None)
    args = parser.parse_args()
    args.stream = int(args.stream)
    if args.dst:
        checkpath(args.dst)

    if args.annpaths is not None and len(args.annpaths) == 1 and args.annpaths[0] == '-':
        args.annpaths = [imgpath.replace('image', 'label') for imgpath in args.imgpaths]

    print(args)

    return args

if __name__ == '__main__':
    args = parse_args()

    imgs_filter = []
    if args.filter is not None:
        with open(args.filter, 'r') as fr:
            lines = [line.strip() for line in fr.readlines()]
            imgs_filter.extend(lines)

    imgpath0 = args.imgpaths[0]
    cnt = 0
    cv2.namedWindow('viewimg', cv2.WINDOW_NORMAL) 

    for dirpath, fpaths in walk(imgpath0, regex=args.regex):
        idx = 0
        if imgs_filter:
            fpaths = [fpath for fpath in fpaths if imgs_filter and fpath in imgs_filter]
        while 1:
            if idx >= len(fpaths):
                break

            cnt += 1
            imgfpath0 = Path(fpaths[idx])
            rel_imgfpath0 = imgfpath0.relative_to(imgpath0)
            print('==> cnt: {:<6d}, idx {:<6d}, abs: {} rel: {}'.format(cnt, idx, imgfpath0, rel_imgfpath0))
            
            imgs = []
            show_tag = True
            for i, imgpath in enumerate(args.imgpaths):                
                imgfpath = imgpath / rel_imgfpath0                 
                assert(imgfpath.exists()), 'imgfpath: {}'.format(imgfpath)
                img = cv2.imread(imgfpath.as_posix())
                if args.annpaths is not None:
                    annfpath = Path(args.annpaths[i]) / rel_imgfpath0.with_suffix('.txt')
                    if annfpath.exists():
                        anns = parse_dets(annfpath.as_posix())
                    else:
                        anns = []
                        print('*** annfpath missing: {} ***'.format(annfpath))
                    if i == 0 and args.classes is not None:
                        anns = [ann for ann in anns if ann['cate'] in args.classes]
                        if i == 0 and len(anns) == 0:
                            show_tag = False
                    draw_bboxes(img, anns, imginfo=imgfpath.name)
                img = cv2.resize(img, (640, 384))
                imgs.append(img)

            if not show_tag:
                idx += 1
                cnt += 1
                continue

            # imgshow
            if len(args.imgpaths) == 1:
                img_show = imgs[0]
            elif len(args.imgpaths) == 2:
                img_show = np.hstack([imgs[0], imgs[1]])
            elif len(args.imgpaths) == 3:
                img_show = np.vstack([np.hstack([imgs[0], imgs[1]]), np.hstack([imgs[2], np.zeros_like(imgs[0])])])
            elif len(args.imgpaths) == 4:
                img_show = np.vstack([np.hstack([imgs[0], imgs[1]]), np.hstack([imgs[2], imgs[3]])])
            else:
                assert(False)
            cv2.imshow('viewimg', img_show)
            if args.dst:
                dst = args.dst / rel_imgfpath0.parent
                checkpath(dst, ok='exist_ok')

            key = cv2.waitKey(args.stream)
            if key == ord('q'):
                sys.exit(0)
            elif key == ord('j'):
                idx += 500
                cnt += 500
            elif key == ord('k'):
                idx -= 500
                cnt -= 500
            elif key == ord('b'):
                idx -= 1
                cnt -= 1
            elif key == ord('s'):
                with open('save.txt', 'a') as fa:
                    fa.write('{}\n'.format(imgfpath0))
                idx += 1
                cnt += 1
            else:
                idx += 1
                cnt += 1

        if args.dst:
            print('\nResults saved to {}\n'.format(Path(args.dst).resolve()))
    
    cv2.destroyAllWindows()        


