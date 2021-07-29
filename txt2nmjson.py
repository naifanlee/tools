import os
import json
import sys

from utils import *

catenms = ['car', 'truck', 'bus', 'pedestrian', 'bicycle', 'cone', 'barrier', 'tsign', 'tlight']

p = sys.argv[1]


bboxes_json = []

for fname in listdir(p):
    fname = fname.strip()
    bbox_json = {
        "Cam_folder": "nm",
        "class": "image",
        "date_folder": "20210303",
        "filename": "frame_vc1_77780_rcb.jpg",
        "raw_filename": "frame_vc1_77780.png",
        "task_SpeedLimitSign": [],
        "task_TrafficLight": [],
        "task_arrow": [],
        "task_barrier": [],
        "task_crossing": [],
        "task_freespace": [],
        "task_lane": [],
        "task_parking": [],
        "task_road_traffic_Sign": [],
        "task_trail": [],
        "task_vehicle": []
    }
    bbox_json['filename'] = fname
    bbox_json['raw_filename'] = fname

    vehicles = []
    fpath = os.path.join(p, fname)
    with open(fpath, 'r') as f:
        bboxes = f.readlines()
        for bbox in bboxes:
            bbox = bbox.strip()

            cateid, x, y, w, h = bbox.split()
            cateid = int(cateid)
            catenm = catenms[cateid]
            x, y, w, h = [float(i) for i in [x, y, w, h]]
            x *= 640
            w *= 640
            y *= 384
            h *= 384

            x -= w / 2
            y -= h / 2

            vehicles.append(
                {
                    "tags": {
                        "class": catenm,
                        "height": h,
                        "point": [
                            [
                                "1650",
                                "473"
                            ],
                            [
                                "1650",
                                "811"
                            ],
                            [
                                "1881",
                                "811"
                            ],
                            [
                                "1881",
                                "473"
                            ]
                        ],
                        "type": "rect",
                        "width": w,
                        "x": x,
                        "y": y
                    }
                }
            )
        
    bbox_json['task_vehicle'] = vehicles
    bboxes_json.append(bbox_json)

with open(os.path.join(p, '../dets.json'), 'w') as fout:
    json.dump(bboxes_json, fout, indent=4)