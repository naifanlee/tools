
import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

from utils import *
import glob
from copy import deepcopy

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgpaths', nargs='+', type=str, required=True)
    parser.add_argument('--annpaths', nargs='+', type=str, default=[])
    parser.add_argument('--dst', default=None)
    parser.add_argument('--regex', type=str, default='*')
    parser.add_argument('--stream', action='store_true')
    parser.add_argument('--scale', type=int, default=1)
    parser.add_argument('--name', type=str, required=True)
    args = parser.parse_args()
    args.stream = int(args.stream)
    args.imgpaths = [os.path.join(imgpath, args.name) for imgpath in args.imgpaths]
    args.annpaths = [os.path.join(annpath, args.name) for annpath in args.annpaths]
    args.dst = os.path.join(args.dst, args.name)
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

def refine_label(labels, labels2, olabels2, imgfpath, img2scale):
    labels_tmp = [list(map(lambda x: round(x), [np.clip(label['xlt'], 0, 639), np.clip(label['ylt'], 0, 387), np.clip(label['xrb'], 0, 639), np.clip(label['yrb'], 0, 387)])) for label in labels]
    labels2_tmp = [list(map(lambda x: round(x), [np.clip(label['xlt'], 0, 639), np.clip(label['ylt'], 0, 387), np.clip(label['xrb'], 0, 639), np.clip(label['yrb'], 0, 387)])) for label in labels2]
    merge_anns, merge_oanns = [], []
    labels2_tmp = np.array(labels2_tmp)
    indexA = []
    indexB = []
    print('labels', labels)
    print('labels2', labels2)
    print('labels_tmp', labels_tmp)
    print('labels2_tmp', labels2_tmp)
    for i, label in enumerate(labels_tmp):
        for j, labelj in enumerate(labels2_tmp):
            taga = abs(labels_tmp[i][0] - labels2_tmp[j][0]) <= 1
            tagb = abs(labels_tmp[i][1] - labels2_tmp[j][1]) <= 1
            tagc = abs(labels_tmp[i][2] - labels2_tmp[j][2]) <= 1
            tagd = abs(labels_tmp[i][3] - labels2_tmp[j][3]) <= 1
            if int(taga) + int(tagb) + int(tagc) + int(tagd) >= 3:
                if catenms_train[int(labels2[j]['cate'])] in ['rider', 'bicycle', 'motorcycle', 'tricycle']:
                    merge_anns.append(labels2[j])
                    merge_oanns.append(olabels2[j])
                    indexA.append(i)
                    indexB.append(j)
                elif labels[i]['cate'] == labels2[j]['cate']:
                    merge_anns.append(labels[i])
                    tmpolabels = olabels2[j]
                    tmpolabels['cate'] = labels[i]['cate']
                    merge_oanns.append(tmpolabels)
                    indexA.append(i)
                    indexB.append(j)
                else:
                    merge_anns.append(labels[i])
                    tmpolabels = olabels2[j]
                    tmpolabels['cate'] = labels[i]['cate']
                    merge_oanns.append(tmpolabels)
                    indexA.append(i)
                    indexB.append(j)
                break          
    print(indexA, indexB)
    for a in sorted(indexA, reverse=True):
        del(labels[a])
    for b in sorted(indexB, reverse=True):
        del(labels2[b])

    # changcheng
    if '/0607/' in imgfpath or '/0507/' in imgfpath:
        print(labels2)
        indexC = []
        labelstmp = deepcopy(labels2)
        for i, label in enumerate(labelstmp):
            if catenms_train[int(label['cate'])] in ['tsign']:
                indexC.append(i)
        for a in sorted(indexC, reverse=True):
            del(labels2[a])

        indexD = []
        labelstmp = deepcopy(labels)
        for i, label in enumerate(labelstmp):
            merge_anns.append(label)
            tmpolabels = {'trackid': ''}
            tmpolabels['cate'] = label['cate']
            tmpolabels['xlt'] = label['xlt'] * img2scale
            tmpolabels['ylt'] = label['ylt'] * img2scale
            tmpolabels['xrb'] = label['xrb'] * img2scale
            tmpolabels['yrb'] = label['yrb'] * img2scale
            merge_oanns.append(tmpolabels)
            indexD.append(i)
        for a in sorted(indexD, reverse=True):
            del(labels[a])
        
        if imgfpath == '/data1/Personal/linaifan/olddata/images/changcheng/0507/2021-04-28-16-31/frame_vc1_14323.bmp':
            merge_anns.append(labels2[-1])
            tmpolabels = {'trackid': ''}
            tmpolabels['cate'] = labels2[-1]['cate']
            tmpolabels['xlt'] = labels2[-1]['xlt'] * img2scale
            tmpolabels['ylt'] = labels2[-1]['ylt'] * img2scale
            tmpolabels['xrb'] = labels2[-1]['xrb'] * img2scale
            tmpolabels['yrb'] = labels2[-1]['yrb'] * img2scale
            merge_oanns.append(tmpolabels)


    # byd
    indexC = []
    labelstmp = deepcopy(labels2)
    for i, label in enumerate(labelstmp):
        if catenms_train[int(label['cate'])] in ['barrier', 'cone', 'tlight', 'pedestrian']:
            merge_anns.append(label)
            tmpolabels = {'trackid': ''}
            tmpolabels['cate'] = label['cate']
            tmpolabels['xlt'] = label['xlt'] * img2scale
            tmpolabels['ylt'] = label['ylt'] * img2scale
            tmpolabels['xrb'] = label['xrb'] * img2scale
            tmpolabels['yrb'] = label['yrb'] * img2scale
            merge_oanns.append(tmpolabels)
            indexC.append(i)
    for a in sorted(indexC, reverse=True):
        del(labels2[a])
    labels2 = []
    assert(len(merge_oanns) == len(merge_anns))
    print(labels, labels2)
    return labels, labels2, merge_anns, merge_oanns

def o2s_atransform(oimg, oanns=None, norm=False, dst_shape=(640, 384), crop=None):
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

    # if crop is None:  # default
    #     if oimg.shape == (1280, 1920, 3):  # Big TDA4
    #         tpcrop, btcrop = 90, 38
    #     elif oimg.shape == (1208, 1920, 3):  # Big PX2
    #         tpcrop, btcrop = 56, 0
    #     elif oimg.shape == (1152, 1920, 3):  # Big Crop Image
    #         tpcrop, btcrop = 0, 0
    #     elif oimg.shape == (384, 640, 3):  # Small Image
    #         tpcrop, btcrop = 0, 0
    #     else:  # HuaWei(1080, 1920)
    #         tpcrop, btcrop = 0, 0
    #         print('  oimg.shape: {} is not right.'.format(oimg.shape))
    # else:
    tpcrop, btcrop = crop

    # sanns, simgs
    sanns = []
    oimg = oimg[tpcrop:oimg.shape[0]-btcrop, :, :]
    oimgh, oimgw = oimg.shape[:2]
    simg = cv2.resize(oimg, dst_shape)
    if oanns is not None:
        oanns, sanns = ann_transform(oanns, (tpcrop, btcrop), oimg.shape, dst_shape)


    # snnas_norm
    sanns_norm = []
    if norm:
        dstw, dsth = dst_shape
        sanns_norm = [[sann[0], sann[1]/dstw, sann[2]/dsth, sann[3]/dstw, sann[4]/dsth] for sann in sanns]
    
    return oimg, oanns, simg, sanns, sanns_norm

def annwrite(annfpath, anns):
    if os.path.exists(annfpath):
        os.system('rm -rf {}'.format(annfpath))
    with open(annfpath, 'a') as fa:
        for ann in anns:
            catenm = catenms_train[int(ann[0])]

            ann = ['cate_x1y1_x2y2', catenm] + ann[1:]
            fa.write(' '.join(list(map(lambda x: str(x), ann))) + '\n')

if __name__ == '__main__':
    args = parse_args()


    imgpath = args.imgpaths[0]
    cnt = 0
    cv2.namedWindow('viewimg', cv2.WINDOW_NORMAL)    
    for dirpath, fpaths in walk(imgpath, regex=args.regex):
        idx = 0

        crop = None
        while 1:
            if idx >= len(fpaths):
                break
            imgfpath = Path(fpaths[idx])
            # if '14323' not in imgfpath.as_posix():
            #     idx += 1
            #     continue


            # coco filter
            if 'coco' in dirpath:
                idx += 1
                continue

            img1_ori = cv2.imread(str(imgfpath))
            if args.scale:
                img1 = cv2.resize(img1_ori.copy(), (img1_ori.shape[1] // args.scale, img1_ori.shape[0] // args.scale))
            cv2.putText(img1, imgfpath.name, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            rel_imgfpath = imgfpath.relative_to(imgpath)
            if len(args.annpaths) >= 1:
                rel_annofpath = rel_imgfpath.with_suffix('.txt').as_posix().replace('images',  'labels')
                annofpath = (Path(args.annpaths[0]) / rel_annofpath).as_posix()
                anns1 = parse_dets(annofpath)
                anns1 = [eval(ann) for ann in list(set([str(ann) for ann in anns1]))]
                cates = [ann['cate'] for ann in anns1]
                for j, cate in enumerate(cates):
                    if cate == '5':
                        anns1[j]['cate'] = '8'
                    if cate == '6':
                        anns1[j]['cate'] = '9'
                    if cate == '7':
                        anns1[j]['cate'] = '10'
                    if cate == '8':
                        anns1[j]['cate'] = '11'
                    if int(cate) >= 9:
                        assert(False), anns1
            cnt += 1
            print('==> cnt: {:<6d}, {}, {}'.format(cnt, imgfpath, annofpath))


            if len(args.imgpaths) >= 2:
                fname = rel_imgfpath.name
                rel_imgpath = rel_imgfpath.parent
                name = fname[:-4].replace('_rcb', '')
                img2fpath = glob.glob('{}/{}/**/{}.*'.format(args.imgpaths[1], rel_imgpath, name), recursive=True)
                if len(img2fpath) == 0:
                    img2fpath = glob.glob('{}/{}/**/{}.*'.format(args.imgpaths[1], rel_imgpath.parent, name), recursive=True)
                assert(len(img2fpath) == 1),"{} {}".format(rel_imgfpath, img2fpath)
                img2fpath = img2fpath[0]
                ann2fpath = glob.glob('{}/{}/**/{}.*'.format(args.annpaths[1], Path(rel_annofpath).parent, name), recursive=True)
                if len(ann2fpath) == 0:
                    ann2fpath = glob.glob('{}/{}/**/{}.*'.format(args.annpaths[1], Path(rel_annofpath).parent.parent, name), recursive=True)
                assert(len(ann2fpath) == 1)
                ann2fpath = ann2fpath[0]
                img2 = cv2.imread(img2fpath)
                if len(args.annpaths) >= 2:
                    print('                ', str(img2fpath), ann2fpath)
                    anns2 = parse_dets(ann2fpath)
                print('anns2', anns2)
                anns2_list = [[ann['cate'], (ann['xlt'] + ann['xrb'])/2, (ann['ylt'] + ann['yrb'])/2, 
                                            ann['xrb'] - ann['xlt'], ann['yrb'] - ann['ylt']] for ann in anns2]
                
                while True:
                    if 'byd_cone/conecase2' in imgfpath.as_posix():
                        crop = [0, 128]
                        break
                    if crop is None:
                        if img2.shape[0] == 1280:
                            crop = [0, 90+38]
                        elif img2.shape[0] == 1208:
                            crop = [0, 56]
                        elif img2.shape[0] == 384:
                            crop = [0, 0]
                        else:
                            print(img2.shape)
                            
                            assert(False)
                    cv2.imwrite('tmp.{}'.format(imgfpath.as_posix()[-3:]), cv2.resize(img2[crop[0]: img2.shape[0]-crop[1], :, :], (640, 384)))
                    img2_jpg = cv2.imread('tmp.{}'.format(imgfpath.as_posix()[-3:]))
                    if np.sum(img1_ori - img2_jpg) != 0:
                        crop = [crop[0] + 1, crop[1] - 1]
                        if crop[1] < 0:
                            print('    can not find crop paras: {}'.format(crop))
                            crop = None
                            assert(False)
                            break
                    else:
                        print('    find crop paras: {}'.format(crop))
                        break
                print('oann', anns2_list)
                anns2_list = [eval(ann) for ann in list(set([str(ann) for ann in anns2_list]))]
                crop_oimg, oanns, simg, sanns, sanns_norm = o2s_atransform(img2, oanns=anns2_list, norm=False, dst_shape=(640, 384), 
                                                                        crop=crop)
                print('oann2', oanns)
                print('sann', sanns)
                sanns = parse_dets(sanns)


            from copy import deepcopy
            oanns_listdict = parse_dets(oanns)
            if crop_oimg.shape[0] == 1152:
                img2scale = 3.
            elif crop_oimg.shape[0] == 384:
                img2scale = 1.
            else:
                assert(False)
            anns1_refine, anns2_refine, merge_anns, merge_oanns = refine_label(deepcopy(anns1), deepcopy(sanns), deepcopy(oanns_listdict), imgfpath.as_posix(), img2scale)
            img1tmp = draw_bboxes(img1.copy(), anns1, show_conf=False, show_text=False)
            img2tmp = draw_bboxes(simg.copy(), sanns, show_conf=False, show_text=False)

            img1_1tmp = draw_bboxes(img1.copy(), anns1_refine, show_conf=False)
            img2_1tmp = draw_bboxes(simg.copy(), anns2_refine, show_conf=False)
            img1show = cv2.addWeighted(img1tmp, 0.5, img1_1tmp, 1, 0)
            img2show = cv2.addWeighted(img2tmp, 0.5, img2_1tmp, 1, 0)
            
            img_merge = draw_bboxes(img1_ori.copy(), merge_anns, show_conf=False)

            # imgshow
            if len(anns1_refine) == len(anns2_refine) == 0:
                args.stream = 1
            else:
                oimg_show = draw_bboxes(deepcopy(crop_oimg), merge_oanns, show_conf=False)
                oimg_show = cv2.resize(oimg_show, (640, 384))
                img_show = np.vstack([np.hstack([img1show, img2show]), np.hstack([img_merge, oimg_show])])
                cv2.imshow('viewimg', img_show)
                args.stream = 0

            if args.dst:
                img2fpath = img2fpath.replace('oimg/', '').replace('simg/', '').split('/')
                print(img2fpath)
                if img2fpath[-3] == img2fpath[-2]:
                    img2fpath = img2fpath[:-2] + img2fpath[-1:]
                print(img2fpath)

                img2fpath = '/'.join(img2fpath)

                ann2fpath = ann2fpath.replace('oimg', '').replace('simg', '').split('/')
                if ann2fpath[-3] == ann2fpath[-2]:
                    ann2fpath = ann2fpath[:-2] + ann2fpath[-1:]
                ann2fpath = '/'.join(ann2fpath)
                imgdst = args.dst / Path('simages') /Path(img2fpath).relative_to(args.imgpaths[1]).parent
                labeldst = args.dst / Path('slabels') / Path(ann2fpath).relative_to(args.annpaths[1]).parent
                oimgdst = args.dst / Path('oimages') /Path(img2fpath).relative_to(args.imgpaths[1]).parent
                olabeldst = args.dst / Path('olabels') / Path(ann2fpath).relative_to(args.annpaths[1]).parent

                checkpath(imgdst, ok='exist_ok')
                checkpath(labeldst, ok='exist_ok')
                checkpath(oimgdst, ok='exist_ok')
                checkpath(olabeldst, ok='exist_ok')

                print((imgdst / rel_imgfpath.with_suffix('.png').name).as_posix())
                print((labeldst / rel_imgfpath.with_suffix('.txt').name).as_posix())
                cv2.imwrite((imgdst / rel_imgfpath.with_suffix('.png').name).as_posix(), simg)
                merge_anns = [[ann['cate'], ann['xlt'], ann['ylt'], ann['xrb'], ann['yrb']] for ann in merge_anns]
                annwrite((labeldst / rel_imgfpath.with_suffix('.txt').name).as_posix(), merge_anns)
                cv2.imwrite((oimgdst / rel_imgfpath.with_suffix('.png').name).as_posix(), crop_oimg)                
                merge_oanns = [[ann['cate'], ann['xlt'], ann['ylt'], ann['xrb'], ann['yrb']] for ann in merge_oanns]
                annwrite((olabeldst / rel_imgfpath.with_suffix('.txt').name).as_posix(), merge_oanns)


            key = cv2.waitKey(args.stream)
            if key == ord('q'):
                sys.exit(0)
            elif key == ord('j'):
                idx += 20
            elif key == ord('k'):
                idx -= 20
            if key == ord('b'):
                idx -= 1
            else:
                idx += 1

        if args.dst:
            print('\nResults saved to {}\n'.format(Path(args.dst).resolve()))
    
    cv2.destroyAllWindows()        


