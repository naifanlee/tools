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
    parser.add_argument('--catenms', nargs='+', type=str, 
                        default=['car', 'truck', 'bus', 'pedestrian', 'bicycle'])
    parser.add_argument('--with-catenm', action='store_true')
    args = parser.parse_args()
    
    args.stream = int(args.stream)
    if args.dst:
        checkpath(args.dst)
    print(args)
    return args

if __name__ == '__main__':
    args = parse_args()
    cv2.namedWindow('viewimg', cv2.WINDOW_NORMAL)

    cnt = 0
    for dirpath, fpaths in walk(args.imgpath, regex='oimages_anno/*.bmp'):
        annjson_fpath = glob.glob(osp.join(dirpath, '../*nm*.json'))[0]
        print('==> Annotation fpath: {}'.format(Path(annjson_fpath).resolve()))
       
        anns_dt = parse_nmjson(annjson_fpath)

        # check dst path
        rel_fpath = Path(dirpath).relative_to(args.imgpath).parent
        dst = Path(args.dst)
        simg_path = dst / 'images' / rel_fpath
        oimg_path = dst / 'oimages' / rel_fpath
        sann_path = dst / 'slabels' / rel_fpath
        oann_path = dst / 'olabels' / rel_fpath
        sann_norm_path = dst / 'slabels_norm' / rel_fpath
        checkpath(simg_path)
        # checkpath(oimg_path)
        # checkpath(sann_path)
        # checkpath(oann_path)
        checkpath(sann_norm_path)

        for oimgfpath in fpaths:
            cnt += 1
            print('  count: {:<7d}, imgfpath: {}'.format(cnt, oimgfpath))
            stem = Path(oimgfpath).stem
            simg_fpath = (simg_path / stem).with_suffix('.png')
            oimg_fpath = (oimg_path / stem).with_suffix('.png')
            sann_fpath = (sann_path / stem).with_suffix('.txt')
            oann_fpath = (oann_path / stem).with_suffix('.txt')
            sann_norm_fpath = (sann_norm_path / stem).with_suffix('.txt')
            oimg = cv2.imread(oimgfpath)
            oanns = anns_dt.get(stem, [])
            oanns = [oann for oann in oanns if oann[0] in args.catenms]
            simg, sanns, sanns_norm = o2s_transform(oimg=oimg, oanns=oanns, norm=True)
            simg_show = draw_bboxes(simg.copy(), sanns)
            cv2.imshow('viewimg', simg_show)
            key = cv2.waitKey(args.stream)
            if key == ord('q'):
                sys.exit(0)
            
            # write_anns(sann_fpath, sanns)
            # write_anns(oann_fpath, oanns)
            def annwrite(annfpath, anns):
                with open(annfpath, 'a') as fa:
                    for ann in anns:
                        ann = [args.catenms.index(ann[0])] + ann[1:]
                        fa.write(' '.join(list(map(lambda x: str(x), ann))) + '\n')
            
            annwrite(sann_norm_fpath.as_posix(), sanns_norm)
            cv2.imwrite(simg_fpath.as_posix(), simg)
            # cv2.imwrite(oimg_fpath, img)                

    cv2.destroyAllWindows()        

