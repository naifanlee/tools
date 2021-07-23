import re
import os.path as osp



def parse_dets(annos):
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
        assert(False)

    for bbox in bboxes:
        if len(bbox) == 5:
            cate = bbox[0]
            xc, yc, w, h = map(float, bbox[1:])
            trackid, conf = '', 0            
        elif len(bbox) == 6:
            trackid = ''
            cate = bbox[0]
            xc, yc, w, h = map(lambda x: int(float(x)), bbox[1:-1])
            conf = round(float(bbox[-1]), 2)           
        elif len(bbox) == 7:
            trackid, cate = bbox[:2]
            xc, yc, w, h = map(lambda x: int(float(x)), bbox[2:-1])
            conf = round(float(bbox[-1]), 2)
        else:
            assert(False), 'len(bbox) must = 5/6/7: {}'.format(bbox)

        xlt = int(xc - w / 2)
        ylt = int(yc - h / 2)
        xrb = int(xc + w / 2)
        yrb = int(yc + h / 2)
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