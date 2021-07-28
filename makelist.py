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
    parser.add_argument('--name', type=str, required=True)
    args = parser.parse_args()

    print(args)
    return args

if __name__ == '__main__':
    args = parse_args()
    for dirpath, fpaths in walk(args.imgpath, regex='**/*.png'):
        for fpath in fpaths:
            with open(args.name + '.txt', 'a') as fa:
                fa.write(fpath + '\n')