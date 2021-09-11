import glob
import sys
from utils import *


dt = {}
cnt = 0
for annfpath in glob.glob('{}/**/slabels/**/*txt'.format(sys.argv[1]), recursive=True):
    cnt += 1
    print(cnt)
    with open(annfpath, 'r') as fr:
        for line in fr.readlines():
            line = line.strip().split()
            cate = catenms_train[int(line[0])]
            if cate in dt:
                dt[cate] += 1
            else:
                dt[cate] = 1


print(dt)
