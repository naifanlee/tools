import argparse
import glob
import os
import os.path as osp
import sys
from pathlib import Path
import json
import cv2

from utils import *

def parse_unmjson(anno_fpath, bicycle_merge=False, tlight_merge=True):
    anns_dt = {}
    for anno in json.load(open(anno_fpath, 'r')):
        anns = []
        for task in ['task_vehicle', 'task_barrier', 'task_road_traffic_Sign', 'task_TrafficLight', 'task_SpeedLimitSign']:
            objs = [obj['tags'] for obj in anno.get(task, [])]
            ''' objs
                {'class': 'pedestrian',
                'height': 696.14,
                'point': [['1308', '403'],
                            ['1308', '1099'],
                            ['1674', '1099'],
                            ['1674', '403']],
                'type': 'rect',
                'width': 366.79,
                'x': 1308.07,
                'y': 403.28}
            '''
            for obj in objs:
                if isinstance(obj['class'], str):
                    catenm = obj['class'].strip()
                elif isinstance(obj['class'], list):  # history problem
                    if isinstance(obj['class'][0], list):
                        catenm = obj['class'][0][0].strip()
                        # print('[[]], not str', anno['raw_filename'].split('/')[-1][:-4], task, obj)
                    else:
                        catenm = '_'.join([cls.strip() for cls in obj['class']])
                        # print('[], not str', anno['raw_filename'].split('/')[-1][:-4], task, obj)
                else:  
                    assert(False)

                if catenm   == 'motocycle':
                    catenm = 'motorcycle'
                    if bicycle_merge:
                        catenm = 'bicycle'
                elif catenm in ['rider']:
                    if bicycle_merge:
                        catenm = 'bicycle'
                # tlight
                elif catenm.endswith('traffic-light'):
                    catenm_tmp = 'tlight'
                    if 'red' in catenm:
                        catenm_tmp += '_red'
                    if 'yellow' in catenm:
                        catenm_tmp += '_yellow'
                    if 'green' in catenm:
                        catenm_tmp += '_green'
                    if 'forward' in catenm:
                        catenm_tmp += '_forward'
                    if 'left' in catenm:
                        catenm_tmp += '_left'
                    if 'right' in catenm:
                        catenm_tmp += '_right'
                    if 'uturn' in catenm:
                        catenm_tmp += '_uturn'
                    if 'bike' in catenm:
                        catenm_tmp += '_bike'
                    if 'pedestrian' in catenm:
                        catenm_tmp += '_pedestrian'
                    catenm = catenm_tmp
                elif catenm.endswith('traffic-sign'):
                    catenm = 'tsign'
                elif catenm in ['plastic-traffic-barrier', 'Safety-Crash-Barrels', 'cement-traffic-barrier']:
                    catenm = 'barrier'
                elif catenm == 'ignore_area':
                    catenm = 'ignore'
                else:
                    if catenm in catenms_train:
                        pass
                    elif task == 'task_SpeedLimitSign':
                        if catenm.startswith('road') or catenm.startswith('ramp') or catenm.startswith('unlimited'):
                            catenm = 'tsign'
                    elif catenm.startswith('back-sign') or catenm.startswith('light_on') or catenm.startswith('inferred-stopline'):
                        continue
                    elif catenm.startswith('pedestrian') or catenm.startswith('bike') \
                        or catenm.startswith('circle') \
                        or catenm.startswith('off-sign')  \
                        or catenm.startswith('left') or catenm.startswith('right')\
                        or catenm.startswith('forward') or catenm.startswith('uturn') \
                        or catenm.startswith('side-sign') \
                        or catenm.startswith('green') or catenm.startswith('red') or catenm.startswith('yellow'):
                        catenm_tmp = 'tlight'
                        if 'red' in catenm:
                            catenm_tmp += '_red'
                        if 'yellow' in catenm:
                            catenm_tmp += '_yellow'
                        if 'green' in catenm:
                            catenm_tmp += '_green'
                        if 'forward' in catenm:
                            catenm_tmp += '_forward'
                        if 'left' in catenm:
                            catenm_tmp += '_left'
                        if 'right' in catenm:
                            catenm_tmp += '_right'
                        if 'uturn' in catenm:
                            catenm_tmp += '_uturn'
                        if 'bike' in catenm:
                            catenm_tmp += '_bike'
                        if 'pedestrian' in catenm:
                            catenm_tmp += '_pedestrian'
                        catenm = catenm_tmp
                    else:
                        if not catenm.startswith('vehicle'):
                            print('catenm, not defined about traffic light', anno['raw_filename'].split('/')[-1][:-4], task, obj)
                        else:
                            print('catenm, not defined about traffic light', anno['raw_filename'].split('/')[-1][:-4], task, obj)
                            catenm = 'car'
                try:
                    xlt, ylt, w, h = list(map(float, [obj['x'], obj['y'], obj['width'], obj['height']]))
                except:
                    # print('point, not rect', anno['raw_filename'].split('/')[-1][:-4], task, obj)
                    continue
            
                anns.append([catenm, xlt + w / 2, ylt + h / 2, w, h])
            
        key = anno['raw_filename'].split('/')[-1][:-4]
        anns_dt[key] = anns
    return anns_dt

def annwrite(annfpath, anns):
    if osp.exists(annfpath):
        os.system('rm -rf {}'.format(annfpath))
    with open(annfpath, 'a') as fa:
        for ann in anns:
            catenm = ann[0]
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
    parser.add_argument('--regex', type=str, default=None)
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

        anns_dt = parse_unmjson(annjson_fpath, bicycle_merge=args.bicycle_merge)
        
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
            
            cnt += 1
            print('  count: {:<7d}, imgfpath: {}'.format(cnt, oimgfpath))    
            # oimg = cv2.imread(oimgfpath)
            # if oimg is None:
            #     print('  None img, count: {:<7d}, imgfpath: {}'.format(cnt, oimgfpath))      
            #     continue

            # oanns = [oann for oann in oanns if oann[0] in args.catenms]
            
            # if args.simg:
            #     oimg, oanns, simg, sanns, sanns_norm = o2s_transform(oimg=oimg, oanns=oanns, norm=True)
            #     img_show = draw_bboxes(simg.copy(), parse_dets(sanns), scale=1)
            # if args.oimg:
            #     img_show = draw_bboxes(oimg.copy(), parse_dets(oanns), scale=3)
            
            # if not args.noshow:
            #     cv2.imshow('viewimg', img_show)
            #     key = cv2.waitKey(args.stream)
            #     if key == ord('q'):
            #         sys.exit(0)

            # save simg, sann
            # if args.simg:
            #     simg_fpath = (simg_path / stem).with_suffix('.png').as_posix()
            #     cv2.imwrite(simg_fpath, simg)
            #     sann_fpath = (sann_path / stem).with_suffix('.txt').as_posix()
            #     annwrite(sann_fpath, sanns)
            #     sann_norm_fpath = (sann_norm_path / stem).with_suffix('.txt').as_posix()
            #     annwrite(sann_norm_fpath, sanns_norm)

            # save oimg, oann
            if args.oimg:
                # oimg_fpath = (oimg_path / stem).with_suffix('.png').as_posix()
                # cv2.imwrite(oimg_fpath, oimg)                
                oann_fpath = (oann_path / stem).with_suffix('.txt').as_posix()
                annwrite(oann_fpath, oanns)

            oanns_tmp = [oann for oann in oanns if oann[0].startswith('tlight_')]
            if len(oanns_tmp):
                with open('olabels_with_tlight_color.txt', 'a') as fa:
                    fa.write('{}\n'.format(oann_fpath))

    cv2.destroyAllWindows()  