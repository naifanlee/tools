import json
import re
import os.path as osp

import cv2
import numpy as np
from numpy.lib.arraysetops import isin

from .cfgs import catenms_train

def parse_dets(annos, annotype=''):
    labels = []
    if isinstance(annos, str):
        if not osp.exists(annos):
            print('AnnoFile: "{}" not exists'.format(annos))
            return []

        with open(annos, 'r') as fr:
            bboxes = [bbox.strip().split() for bbox in fr.readlines()]
    elif isinstance(annos, list):
        bboxes = annos
    elif isinstance(annos, dict):
        bboxes = [bbox for _, bbox in annos.items()]
    else:
        assert(False), 'annos.type: {}'.format(type(annos))

    for bbox in bboxes:
        if bbox[0] == 'cate_x1y1_x2y2':
            cate = bbox[1]
            xlt, ylt, xrb, yrb = map(float, bbox[2:])
            trackid, conf = '', 0
        else:
            if len(bbox) == 5:
                cate = bbox[0]
                xc, yc, w, h = map(float, bbox[1:])
                trackid, conf = '', 0            
            elif len(bbox) == 6:
                trackid = ''
                cate = bbox[0]
                xc, yc, w, h = map(lambda x: float(x), bbox[1:-1])
                conf = round(float(bbox[-1]), 2)           
            elif len(bbox) == 7:
                trackid, cate = bbox[:2]
                xc, yc, w, h = map(lambda x: float(x), bbox[2:-1])
                conf = round(float(bbox[-1]), 2)
            else:
                assert(False), 'len(bbox) must = 5/6/7: {}'.format(bbox)

            xlt = xc - w / 2
            ylt = yc - h / 2
            xrb = xc + w / 2
            yrb = yc + h / 2
        cate = 'bicycle' if cate == 'bike' else cate

        labels.append({
            'trackid': trackid,
            'cate': cate,
            'xlt': xlt,
            'ylt': ylt,
            'xrb': xrb,
            'yrb': yrb,
            'conf': conf
        })

    return labels

def parse_nmjson(anno_fpath, bicycle_merge=False, tlight_merge=True):
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
                    catenm = 'tlight'
                elif catenm.endswith('traffic-sign'):
                    catenm = 'tsign'
                elif catenm in ['plastic-traffic-barrier', 'Safety-Crash-Barrels', 'cement-traffic-barrier', 'safety-crash-barrels']:
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
                        catenm = 'tlight'
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

def yuv2img(yuv_fpath, img_fpath, platform='tda4', img_format='png'):
    if platform == 'tda4':
        resolution = '1920x1280'
    elif platform == 'px2':
        resolution = '1920x1208'
    else:
        assert(False)


    if img_format == 'bmp':
        command = "ffmpeg -s {} -pix_fmt nv12 -i {} -pix_fmt rgb24 {}".format(resolution, yuv_fpath, img_fpath)
    elif img_format == 'png':
        command = "ffmpeg -s {} -pix_fmt nv12 -i {} -vframes 1 -compression_level 5 {}".format(resolution, yuv_fpath, img_fpath)
    else:
        assert(False)

    for _ in range(3):
        import os
        error_code = os.system(command)
        if error_code == 0:
            break

    if error_code != 0:
        return error_code

    # img = cv2.imread(img_fpath)

    # # os.remove(bmp_fpath)
    # if img.shape[0] == 1280:
    #     img = img[90:-38, :, :]
    # elif img.shape[0] == 1208:
    #     img = img[56:, :, :]
    # cv2.imwrite(img_fpath, img)

    return error_code

def o2s_transform(oimg, oanns=None, norm=False, dst_shape=(640, 384), crop=None):
    def ann_transform(oanns, crop, src_shape, dst_shape):
        xscale, yscale = src_shape[1] / dst_shape[0], src_shape[0] / dst_shape[1]
        oanns_transform, sanns = [], []
        for oann in oanns:
            cate, xc, yc, w, h = oann
            tpcrop, btcrop = crop
            if tpcrop > 0:
                ylt = max(0, yc - h/2 - (tpcrop-1))
                if yc - h/2 >= tpcrop-1:
                    h = h
                else:
                    if yc + h/2 <= tpcrop-1:
                        continue
                    else:
                        h = h - (tpcrop-1 - (yc - h/2))
                yrb = min(1280-1-btcrop, ylt + h)
                h = yrb - ylt
                yc = (ylt + yrb) / 2
            else:
                yc += abs(tpcrop)
            oanns_transform.append([cate, xc, yc, w, h])
            sanns.append([cate, xc / xscale, yc / yscale, w / xscale, h / yscale])
        return oanns_transform, sanns

    if crop is None:  # default
        if oimg.shape == (1280, 1920, 3):  # Big TDA4
            tpcrop, btcrop = 90, 38
        elif oimg.shape == (1208, 1920, 3):  # Big PX2
            tpcrop, btcrop = 56, 0
        elif oimg.shape == (1152, 1920, 3):  # Big Crop Image
            tpcrop, btcrop = 0, 0
        elif oimg.shape == (384, 640, 3):  # Small Image
            tpcrop, btcrop = 0, 0
        else:  # HuaWei(1080, 1920)
            tpcrop, btcrop = 0, 0
            print('  oimg.shape: {} is not right.'.format(oimg.shape))
    else:
        tpcrop, btcrop = crop

    # sanns, simgs
    sanns = []
    oimg = oimg[tpcrop:oimg.shape[0]-btcrop, :, :]
    oimgh, oimgw = oimg.shape[:2]
    if oimgh == 1152 and oimgw == 1920:
        simg = cv2.resize(oimg, dst_shape)
        if oanns is not None:
            oanns, sanns = ann_transform(oanns, (tpcrop, btcrop), oimg.shape, dst_shape)
    else:
        simg = oimg.copy()
        dst_shape = (oimg.shape[1], oimg.shape[0])
        oanns, sanns = ann_transform(oanns, (tpcrop, btcrop), oimg.shape, dst_shape)

    # snnas_norm
    sanns_norm = []
    if norm:
        dstw, dsth = dst_shape
        sanns_norm = [[sann[0], sann[1]/dstw, sann[2]/dsth, sann[3]/dstw, sann[4]/dsth] for sann in sanns]
    
    return oimg, oanns, simg, sanns, sanns_norm
