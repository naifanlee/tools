import cv2
import random

from .helper_funs import parse_dets
from .utils import clip


random.seed(random.randint(0, 100))
cates = ['car', 'truck', 'bus', 'pedestrian', 'bicycle', 'tricycle', 'tlight', 'tsign', 'ignore']
cate_colors = [[random.randint(0, 255) for _ in range(3)] for cate in cates]

def draw_bboxes(img, label_fpath, imginfo='', show_conf=False):
    def compute_puttext_loc(imgw, basepoint, tsize):
        xlt, ylt, xrb, yrb = basepoint
        if ylt - tsize[-1] - 1 < 0:  # top truncated case
            ylt = max(ylt, 0) + tsize[-1]
        else:
            ylt -= 1
        if xlt + tsize[0] > imgw:

            xlt = min(imgw, xrb) - tsize[0]
            
        return (xlt, ylt)
    
    labels = parse_dets(label_fpath)
    if len(labels) == 0:
        return img

    for label in labels:
        cate, xlt, ylt, xrb, yrb = label['cate'], label['xlt'], label['ylt'], label['xrb'], label['yrb']
        
        # draw bbox
        cate_color = cate_colors[cates.index(cate)]
        cv2.rectangle(img, (xlt, ylt), (xrb, yrb), cate_color, 1)
        
        # putText
        text = ''
        cate = cate[0] if cate in ['car', 'truck'] else cate[:3]
        if label['trackid']:
            text += label['trackid']
        text += cate
        if show_conf and label['conf']:
            print(label['conf'], label['conf'] * 10, round(label['conf'] * 10))
            text += str(round(label['conf'] * 10))
        tsize, baseline = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, thickness=1)
        xlt, ylt = compute_puttext_loc(img.shape[1], (xlt, ylt, xrb, yrb), tsize)
        cv2.rectangle(img, (xlt, ylt), (xlt + tsize[0], ylt - tsize[-1]), cate_color, -1)
        cv2.putText(img, text, (xlt, ylt), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    if imginfo:
        cv2.putText(img, str(imginfo), (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # return img