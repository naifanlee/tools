import argparse
import glob
import os
import os.path as osp
import sys
from pathlib import Path

import cv2

from utils import *


def annwrite(annfpath, anns):
    with open(annfpath, 'a') as fa:
        for ann in anns:
            if args.with_catenm:
                catenm = ann[0]
            else:
                if args.catenms == 'roaduser':
                    catenm = roaduser_5cls.index(ann[0])
                elif args.catenms == 'all':
                    catenm = catenms_train.index(ann[0])
                else:
                    catenm = 'unknown'

            ann = [catenm] + ann[1:]
            fa.write(' '.join(list(map(lambda x: str(x), ann))) + '\n')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgpath', type=str, required=True)
    parser.add_argument('--dst', type=str, default=None)
    parser.add_argument('--stream', action='store_true')
    parser.add_argument('--catenms', nargs='+', type=str, default=None)
    parser.add_argument('--with-catenm', action='store_true')
    parser.add_argument('--oimg', action='store_true')
    parser.add_argument('--regex', type=str, default=None)
    parser.add_argument('--bicycle-merge', action='store_true')
    args = parser.parse_args()
    
    args.stream = int(args.stream)
    if args.dst:
        checkpath(args.dst)
    print(args)
    return args

if __name__ == '__main__':
    args = parse_args()
    if not args.stream:
        cv2.namedWindow('viewimg', cv2.WINDOW_NORMAL)

    cnt = 0
    for dirpath, fpaths in walk(args.imgpath, args.regex):
        annjson_fpath = glob.glob(osp.join(osp.dirname(dirpath), '**/*.json'), recursive=True)
        if len(annjson_fpath) == 1:
            annjson_fpath = annjson_fpath[0]
            print('==> Annotation fpath: {}'.format(Path(annjson_fpath).resolve()))
        else:
            assert(False), '{} {}'.format(dirpath, annjson_fpath)
        anns_dt = parse_nmjson(annjson_fpath, bicycle_merge=args.bicycle_merge)
        continue
        # check dst path
        rel_fpath = Path(dirpath).relative_to(args.imgpath).parent
        dst = Path(args.dst)
        simg_path = dst / 'simages' / rel_fpath
        checkpath(simg_path)
        
        if args.oimg:
            oimg_path = dst / 'oimages' / rel_fpath
            checkpath(oimg_path)
            oann_path = dst / 'olabels' / rel_fpath
            checkpath(oann_path)

        sann_path = dst / 'slabels' / rel_fpath
        checkpath(sann_path)
        sann_norm_path = dst / 'slabels_norm' / rel_fpath
        checkpath(sann_norm_path)

        for oimgfpath in fpaths:
            stem = Path(oimgfpath).stem
            oanns = anns_dt.get(stem, [])
            
            # # filter cone
            # cone_tag = False
            # for oann in oanns:
            #     if oann[0] == 'cone':
            #         cone_tag = True
            # if cone_tag == False:
            #     continue
            
            cnt += 1
            print('  count: {:<7d}, imgfpath: {}'.format(cnt, oimgfpath))    
            oimg = cv2.imread(oimgfpath)

            if args.catenms is not None:
                if args.catenms == 'roaduser':
                    oanns = [oann for oann in oanns if oann[0] in roaduser_cls]
                if args.catenms == 'all':
                    oanns = [oann for oann in oanns if oann[0] in catenms_train]


            oimg, oanns, simg, sanns, sanns_norm = o2s_transform(oimg=oimg, oanns=oanns, norm=True)
            simg_show = draw_bboxes(simg.copy(), parse_dets(sanns))
            if not args.stream:
                cv2.imshow('viewimg', simg_show)
            # oimg_show = draw_bboxes(oimg.copy(), oanns)
            # cv2.imshow('viewimg', oimg_show)
                key = cv2.waitKey(args.stream)
                if key == ord('q'):
                    sys.exit(0)

            # save simg, sann
            simg_fpath = (simg_path / stem).with_suffix('.png')
            cv2.imwrite(simg_fpath.as_posix(), simg)
            sann_fpath = (sann_path / stem).with_suffix('.txt')
            annwrite(sann_fpath.as_posix(), sanns)
            sann_norm_fpath = (sann_norm_path / stem).with_suffix('.txt')
            annwrite(sann_norm_fpath.as_posix(), sanns_norm)

            # save oimg, oann
            if args.oimg:
                oimg_fpath = (oimg_path / stem).with_suffix('.png')
                cv2.imwrite(oimg_fpath, oimg)                
                oann_fpath = (oann_path / stem).with_suffix('.txt')
                annwrite(oann_fpath.as_posix(), oanns)

            
    cv2.destroyAllWindows()  