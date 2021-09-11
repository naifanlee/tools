# 不同分辨率比较
import argparse
import os
import sys
from pathlib import Path
from copy import deepcopy as cp

import cv2
import numpy as np

from utils import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgpath', nargs='+', type=str, required=True)
    parser.add_argument('--annopath', nargs='+', type=str, default=None)
    parser.add_argument('--dst', default=None)
    parser.add_argument('--regex', type=str, default='*.bmp')
    parser.add_argument('--stream', action='store_true')
    args = parser.parse_args()
    args.stream = int(args.stream)
    if args.dst:
        checkpath(args.dst)
    print(args)

    return args

def calcu_iou(bboxA, bboxesB):
    width = np.minimum(bboxA[2], bboxesB[:, 2]) - np.maximum(bboxA[0], bboxesB[:, 0])
    height = np.minimum(bboxA[3], bboxesB[:, 3]) - np.maximum(bboxA[1], bboxesB[:, 1])
    Aarea = (bboxA[2] - bboxA[0]) * (bboxA[3] - bboxA[1])
    Barea = (bboxesB[:, 2] - bboxesB[:, 0]) * (bboxesB[:, 3] - bboxesB[:, 1])
    iner_area = width * height
    ious = iner_area / (Aarea + Barea - iner_area)
    ious[width < 0] = 0.0
    ious[height < 0] = 0.0
    return ious

def refine_label(labels, labels2):
    labels_tmp = [[label['xlt'], label['ylt'], label['xrb'], label['yrb']] for label in labels]
    labels2_tmp = [[label['xlt'], label['ylt'], label['xrb'], label['yrb']] for label in labels2]
    labels2_tmp = np.array(labels2_tmp)
    indexA = []
    indexB = []
    for i, label in enumerate(labels_tmp):
        ious = calcu_iou(np.array(label), labels2_tmp)
        if np.max(ious) > 0.8:
            index = int(np.argmax(ious))
            indexA.append(i)
            indexB.append(index)
    
    for a in sorted(indexA, reverse=True):
        del(labels[a])
    for b in sorted(indexB, reverse=True):
        del(labels2[b])
            
    return labels, labels2


if __name__ == '__main__':
    args = parse_args()


    imgpath = args.imgpath[0]
    # cv2.namedWindow('viewimg', cv2.WINDOW_NORMAL)    
    for dirpath, fpaths in walk(imgpath, regex=args.regex):
        idx = 0
        while 1:
            imgfpath = Path(fpaths[idx])
            print('idx: {:<6d}, {}'.format(idx, imgfpath))

            rel_fpath = imgfpath.relative_to(imgpath)
            fname = imgfpath.name
            if args.annopath:
                rel_annofpath = rel_fpath.with_suffix('.txt').as_posix().replace('/images/',  '/labels/')
                annofpath = (Path(args.annopath[0]) / rel_annofpath).as_posix()
                labels = parse_dets(annofpath)  

            img1 = cv2.imread(str(imgfpath))
            if len(args.imgpath) == 1:
                img = img1
            elif len(args.imgpath) == 2:
                imgfpath2 = args.imgpath[1] / rel_fpath
                if args.annopath:
                    annofpath2 = (Path(args.annopath[1]) / rel_annofpath).as_posix()
                    labels2 = parse_dets(annofpath2)  
                img2 = cv2.imread(imgfpath2.as_posix())
                
                # labels, labels2 = refine_label(cp(labels), cp(labels2))
                draw_bboxes(img2, labels2, show_conf=True)
                
                h1, w1 = img1.shape[:2]
                h2, w2 = img2.shape[:2]
                h = max(h1, h2)
                if h1 < h:
                    img1 = np.pad(img1, ((h-h1, 0), (0,0), (0,0)), mode='constant')
                if h2 < h:
                    img2 = np.pad(img2, ((h-h2, 0), (0,0), (0,0)), mode='constant')
                img = np.vstack([img1, img2])
            else:
                assert(False)

            if args.annopath:
               draw_bboxes(img, labels, show_conf=True)
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
            elif key == ord('j'):
                idx += 100
            elif key == ord('k'):
                idx -= 100
            else:
                idx += 1
            
            if idx + 1 > len(fpaths):
                break

        if args.dst:
            print('\nResults saved to {}\n'.format(Path(args.dst).resolve()))
    
    cv2.destroyAllWindows() 