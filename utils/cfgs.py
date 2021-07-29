import random

random.seed(random.randint(0, 100))
roaduser_5cls = ['car', 'truck', 'bus', 'pedestrian', 'bicycle']
catenms_train = ['car', 'truck', 'bus', 'pedestrian', 'bicycle', 'motorcycle', 'tricycle',
            'rider', 'cone', 'barrier', 'tsign', 'tlight']
catenms_test = ['car', 'truck', 'bus', 'pedestrian', 'bicycle', 'motorcycle', 'tricycle',
            'rider', 'cone', 'barrier', 'tsign', 'tlight', 'ignore']
cate_colors = [[random.randint(0, 255) for _ in range(3)] for cate in catenms_test]