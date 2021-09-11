import argparse
import glob
import os
import os.path as osp
import sys
from pathlib import Path

import cv2

from utils import *


def annwrite(annfpath, anns):
    if osp.exists(annfpath):
        os.system('rm -rf {}'.format(annfpath))
    with open(annfpath, 'a') as fa:
        for ann in anns:
            if args.with_catenm:
                catenm = ann[0]
            else:
                catenm = catenms_train.index(ann[0])

            ann = [catenm] + ann[1:]
            fa.write(' '.join(list(map(lambda x: str(x), ann))) + '\n')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgpath', type=str, required=True)
    parser.add_argument('--dst', type=str, default=None)
    parser.add_argument('--stream', action='store_true')
    parser.add_argument('--catenms', nargs='+', type=str, default=['car', 'truck', 'bus', 'pedestrian', 
                        'bicycle', 'motorcycle', 'tricycle', 'rider', 'cone', 'barrier', 'tsign', 'tlight'])
    parser.add_argument('--with-catenm', action='store_true')
    parser.add_argument('--oimg', action='store_true')
    parser.add_argument('--simg', action='store_true')
    parser.add_argument('--regex', type=str, default='**/oimg/**/*')
    parser.add_argument('--bicycle-merge', action='store_true')
    parser.add_argument('--noshow', action='store_true')
    args = parser.parse_args()
    
    args.stream = int(args.stream)
    if args.dst:
        checkpath(args.dst)
    print(args)
    return args

if __name__ == '__main__':
    args = parse_args()
    if not args.noshow:
        cv2.namedWindow('viewimg', cv2.WINDOW_NORMAL)

    cnt = 0
    for dirpath, fpaths in walk(args.imgpath, args.regex):
        print('==> Dirpath: {}'.format(Path(dirpath).resolve()))
        
        # Load annotation json file
        annjson_fpath = glob.glob(osp.join(osp.dirname(dirpath), '**/*.json'), recursive=True)
        if len(annjson_fpath) == 1:
            annjson_fpath = annjson_fpath[0]
        elif len(annjson_fpath) == 0:
            annjson_fpath = glob.glob(osp.join(osp.dirname(osp.dirname(dirpath)), '**/*.json'), recursive=True)[0]
        else:
            assert(False)
        print('    Annotation fpath: {}'.format(Path(annjson_fpath).resolve()))

        anns_dt = parse_nmjson(annjson_fpath, bicycle_merge=args.bicycle_merge)
        # Check dst path: simg, sann, sann_norm, oimg, oann
        rel_fpath = Path(dirpath).relative_to(args.imgpath).parent
        dst = Path(args.dst)
        if args.simg:
            simg_path = dst / 'simages' / rel_fpath
            checkpath(simg_path)
            sann_path = dst / 'slabels' / rel_fpath
            checkpath(sann_path)
            sann_norm_path = dst / 'slabels_norm' / rel_fpath
            checkpath(sann_norm_path)

        if args.oimg:
            oimg_path = dst / 'oimages' / rel_fpath
            checkpath(oimg_path)
            oann_path = dst / 'olabels' / rel_fpath
            checkpath(oann_path)


        for oimgfpath in fpaths:
            stem = Path(oimgfpath).stem
            oanns = anns_dt.get(stem, [])
            if not len(oanns):
                oanns = anns_dt.get(stem.replace('_rcb', ''), [])
            
            cnt += 1
            print('  count: {:<7d}, imgfpath: {}'.format(cnt, oimgfpath))    
            oimg = cv2.imread(oimgfpath)
            if oimg is None:
                print('  None img, count: {:<7d}, imgfpath: {}'.format(cnt, oimgfpath))      
                continue

            oanns = [oann for oann in oanns if oann[0] in args.catenms]
            
            if args.simg:
                oimg, oanns, simg, sanns, sanns_norm = o2s_transform(oimg=oimg, oanns=oanns, norm=True)
                img_show = draw_bboxes(simg.copy(), parse_dets(sanns))
            if args.oimg:
                img_show = draw_bboxes(oimg.copy(), parse_dets(oanns))
            
            if not args.noshow:
                cv2.imshow('viewimg', img_show)
                key = cv2.waitKey(args.stream)
                if key == ord('q'):
                    sys.exit(0)

            # save simg, sann
            if args.simg:
                simg_fpath = (simg_path / stem).with_suffix('.png').as_posix()
                cv2.imwrite(simg_fpath, simg)
                sann_fpath = (sann_path / stem).with_suffix('.txt').as_posix()
                annwrite(sann_fpath, sanns)
                sann_norm_fpath = (sann_norm_path / stem).with_suffix('.txt').as_posix()
                annwrite(sann_norm_fpath, sanns_norm)

            # save oimg, oann
            if args.oimg:
                oimg_fpath = (oimg_path / stem).with_suffix('.png').as_posix()
                cv2.imwrite(oimg_fpath, oimg)                
                oann_fpath = (oann_path / stem).with_suffix('.txt').as_posix()
                annwrite(oann_fpath, oanns)

            
    cv2.destroyAllWindows()  