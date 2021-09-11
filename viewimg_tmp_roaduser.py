
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
    labels_tmp = [list(map(lambda x: round(x), [np.clip(label['xlt'], 0, 639), np.clip(label['ylt'], 0, 383), np.clip(label['xrb'], 0, 639), np.clip(label['yrb'], 0, 383)])) for label in labels]
    labels2_tmp = [list(map(lambda x: round(x), [np.clip(label['xlt'], 0, 639), np.clip(label['ylt'], 0, 383), np.clip(label['xrb'], 0, 639), np.clip(label['yrb'], 0, 383)])) for label in labels2]
    merge_anns, merge_oanns = [], []
    labels2_tmp = np.array(labels2_tmp)
    indexA = []
    indexB = []
    print('labels', labels)
    print('labels2', labels2)
    print('olabels', olabels2)
    print('labels_tmp', labels_tmp)
    print('labels2_tmp', labels2_tmp)
    for i, label in enumerate(labels_tmp):
        for j, labelj in enumerate(labels2_tmp):
            taga = abs(labels_tmp[i][0] - labels2_tmp[j][0]) <= 1
            tagb = abs(labels_tmp[i][1] - labels2_tmp[j][1]) <= 1
            tagc = abs(labels_tmp[i][2] - labels2_tmp[j][2]) <= 1
            tagd = abs(labels_tmp[i][3] - labels2_tmp[j][3]) <= 1
            if int(taga) + int(tagb) + int(tagc) + int(tagd) >= 3:
                if j in indexB:
                    continue
                if catenms_train[int(labels2[j]['cate'])] in ['rider', 'bicycle', 'motorcycle', 'tricycle']:
                    merge_anns.append(labels2[j])
                    merge_oanns.append(olabels2[j])
                    indexA.append(i)
                    indexB.append(j)
                elif labels[i]['cate'] == labels2[j]['cate']:
                    print(i, j, labels[i], olabels2[j])
                    merge_anns.append(labels[i])
                    merge_oanns.append(olabels2[j])
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

    if len(labels2) != 0 and len(labels) >= 2 and 'hardsample' not in imgfpath and 'lingang' not in imgfpath:
        assert(len(labels) != len(labels_tmp)), '{} {}'.format(len(labels), len(labels_tmp))
        
    # print(len(labels), len(labels_tmp))
    # assert(len(labels) != len(labels_tmp) and len(labels_tmp) != 0)
    indexS = []
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
        indexS.append(i)
    for a in sorted(indexS, reverse=True):
        del(labels[a])


    if 'ground1119' in imgfpath:
        indexC = []
        labelstmp = deepcopy(labels2)
        for i, label in enumerate(labelstmp):
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

    if 'lg0121' in imgfpath or 'lingang' in imgfpath or 'lg1227' in imgfpath:
        indexD = []
        labelstmp = deepcopy(labels)
        for i, label in enumerate(labelstmp):
            if catenms_train[int(label['cate'])] in ['tsign']:
                merge_anns.append(label)
                tmpolabels = {'trackid': ''}
                tmpolabels['cate'] = label['cate']
                tmpolabels['xlt'] = label['xlt'] * img2scale
                tmpolabels['ylt'] = label['ylt'] * img2scale
                tmpolabels['xrb'] = label['xrb'] * img2scale
                tmpolabels['yrb'] = label['yrb'] * img2scale
                merge_oanns.append(tmpolabels)
                indexD.append(i)
        labels2 = []
        for a in sorted(indexD, reverse=True):
            del(labels[a])
    if 'lg0121' in imgfpath or 'lg1227' in imgfpath or 'lingang' in imgfpath:
        indexC = []
        labelstmp = deepcopy(labels2)
        for i, label in enumerate(labelstmp):
            if catenms_train[int(label['cate'])] in ['tlight']:
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

    if 'tianjin_desay' in imgfpath:
        indexD = []
        labelstmp = deepcopy(labels2)
        for i, label in enumerate(labelstmp):
            indexD.append(i)
        for a in sorted(indexD, reverse=True):
            del(labels2[a])

    if 'tda4_avp_1231' in imgfpath or 'hardsample' in imgfpath:
        indexC = []
        labelstmp = deepcopy(labels2)
        for i, label in enumerate(labelstmp):
            merge_anns.append(label)
            tmpolabels = {'trackid': ''}
            tmpolabels['cate'] = label['cate']
            tmpolabels['xlt'] = label['xlt'] * img2scale
            tmpolabels['ylt'] = label['ylt'] * img2scale
            tmpolabels['xrb'] = label['xrb'] * img2scale
            tmpolabels['yrb'] = label['yrb'] * img2scale
            merge_oanns.append(tmpolabels)
            indexC.append(i)
        labels2 = []

    if 'changtai' in imgfpath:
        indexD = []
        labelstmp = deepcopy(labels2)
        for i, label in enumerate(labelstmp):
            if catenms_train[int(label['cate'])] in ['tlight']:
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
            del(labels2[a])
        labels2 = []
    labels2 = []
    # roaduser
    assert(len(merge_oanns) == len(merge_anns))
    print(labels, labels2, merge_oanns)
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
    print(oimg.shape)
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
    print(oimg.shape)
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
                print(1, '{}/{}/**/{}.*'.format(args.imgpaths[1], rel_imgpath, name))
                if len(img2fpath) == 0:
                    img2fpath = glob.glob('{}/{}/**/{}.*'.format(args.imgpaths[1], rel_imgpath.parent, name), recursive=True)
                if len(img2fpath) == 0:
                    img2fpath = glob.glob('{}/{}/**/{}.*'.format(args.imgpaths[1], rel_imgpath, fname[:-4]), recursive=True)
                    name = fname[:-4]
                if len(img2fpath) == 0:
                    img2fpath = glob.glob('{}/{}/**/{}.*'.format(args.imgpaths[1], rel_imgpath.parent, fname[:-4]), recursive=True)
                    name = fname[:-4]
                # assert(len(img2fpath) == 1),"{} {}".format(rel_imgfpath, img2fpath)
                if len(img2fpath) != 1:
                    print('\n\n ******************')
                    print('{}/{}/**/{}.*'.format(args.imgpaths[1], rel_imgpath, name))
                    print('\n\n ******************')
                    idx += 1
                    continue
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
                anns2_list = [[ann['cate'], (ann['xlt'] + ann['xrb'])/2, (ann['ylt'] + ann['yrb'])/2, 
                                            ann['xrb'] - ann['xlt'], ann['yrb'] - ann['ylt']] for ann in anns2]
                

                if 'byd_cone/conecase2' in imgfpath.as_posix():
                    crop = [0, 128]
                    break
                if img2.shape[0] == 1280:
                    crop = [0, 90+38]
                elif img2.shape[0] == 1208:
                    crop = [56, 0]
                elif img2.shape[0] == 384:
                    crop = [0, 0]
                elif img2.shape[0] == 960:
                    crop = [42, 150]
                elif img2.shape[0] == 1156:
                    crop = [4, 0]    
                else:
                    print(img2.shape)
                    assert(False)
                anns2_list = [eval(ann) for ann in list(set([str(ann) for ann in anns2_list]))]
                crop_oimg, oanns, simg, sanns, sanns_norm = o2s_atransform(img2, oanns=anns2_list, norm=False, dst_shape=(640, 384), 
                                                                    crop=crop)
                sanns = parse_dets(sanns)
                    # if len(sanns)>=1 and len(anns1)>=1:
                    #     cate1, xlt1, ylt1, xrb1, yrb1 = anns1[0]['cate'], anns1[0]['xlt'], anns1[0]['ylt'], anns1[0]['xrb'], anns1[0]['yrb']
                    #     cate2, xlt2, ylt2, xrb2, yrb2 = sanns[0]['cate'], sanns[0]['xlt'], sanns[0]['ylt'], sanns[0]['xrb'], sanns[0]['yrb']
                    #     taga = abs(xlt1 - xlt2) <= 1
                    #     tagb = abs(ylt1 - ylt2) <= 1
                    #     tagc = abs(xrb1 - xrb2) <= 1
                    #     tagd = abs(yrb1 - yrb2) <= 1
                    #     print(crop)
                    #     if int(taga) + int(tagb) + int(tagc) + int(tagd) >= 3 and cate1 == cate2:
                    #         print('    find crop paras: {}'.format(crop))
                    #         break
                    #     else:
                    #         crop = [crop[0] + 1, crop[1] - 1]
                    #         if crop[1] < 0:
                    #             print('    can not find crop paras: {}'.format(crop))
                    #             crop = None
                    #             assert(False)
                    #             break
                        
            from copy import deepcopy
            oanns_listdict = parse_dets(oanns)
            if crop_oimg.shape[0] == 1152:
                img2scale = 3.
            elif crop_oimg.shape[0] == 384:
                img2scale = 1.
            elif crop_oimg.shape[0] == 384*2:
                img2scale = 2.
            else:
                assert(False), crop_oimg.shape
            anns1_refine, anns2_refine, merge_anns, merge_oanns = refine_label(deepcopy(anns1), deepcopy(sanns), deepcopy(oanns_listdict), imgfpath.as_posix(), img2scale=img2scale)
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


