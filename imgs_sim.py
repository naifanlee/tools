import argparse
from PIL import Image

import cv2
import numpy as np
import torch
from torch import nn
from torchvision import transforms
from sklearn.metrics import mutual_info_score

from utils import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgpath', type=str, required=True)
    parser.add_argument('--dst', type=str, default=None)
    parser.add_argument('--stream', action='store_true')
    parser.add_argument('--regex', type=str, default=None)
    args = parser.parse_args()
    
    args.stream = int(args.stream)
    if args.dst:
        checkpath(args.dst)
    print(args)
    return args

class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)



def load(imgfpath):
    with open(imgfpath, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

if __name__ == '__main__':
    args = parse_args()
    cv2.namedWindow('viewimg', cv2.WINDOW_NORMAL)

    imgfpaths = []
    for dirpath, dir_fpaths in walk(args.imgpath, args.regex):
        imgfpaths.extend(dir_fpaths)

    imgfpaths = imgfpaths[10000:150000]
    
    compute_ssim = SSIM()
    idx = 0
    while 1:
        imgfpath1 = imgfpaths[idx]
        print(imgfpath1)
        img1 = transforms.ToTensor()(load(imgfpath1)).unsqueeze(0)
        if idx >= len(imgfpaths) - 1:
            break

        FPS = 40
        tmp_imgfpaths = {}
        for i in range(1, 9):
            tmp_imgfpaths[i] = imgfpaths[idx + i*FPS]


        # ssim0 = torch.mean(compute_ssim(img1, transforms.ToTensor()(load(imgfpaths[idx+1])).unsqueeze(0))).numpy() * 255.
        # ssim0 = round(ssim0, 2)
        # metrics = []
        # for seq, tmp_imgfpath in tmp_imgfpaths.items():
        #     tmp_img = transforms.ToTensor()(load(tmp_imgfpath)).unsqueeze(0)
        #     ssim = torch.mean(compute_ssim(img1, tmp_img)).numpy() * 255.
        #     metrics.append(str(ssim0) + ' ' + str(round(ssim, 2)) + ' ' + str(round((ssim - ssim0) / ssim0, 2)))


        mu0 = mutual_info_score(img1, cv2.imread(imgfpaths[idx + 1]))
        metrics = []
        for seq, tmp_imgfpath in tmp_imgfpaths.items():
            tmp_img = cv2.imread(tmp_imgfpath)
            mu = mutual_info_score(img1, tmp_img)
            metrics.append(str(mu0) + ' ' + str(round(mu, 2)) + ' ' + str(round((mu - mu0) / mu0, 2)))


        img0 = cv2.imread(imgfpath1)
        cv2.putText(img0, imgfpath1.split('/')[-1], (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        img1 = cv2.imread(tmp_imgfpaths[1])
        cv2.putText(img1, tmp_imgfpaths[1].split('_')[-1] + '  ' + metrics[0], (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        img2 = cv2.imread(tmp_imgfpaths[2])
        cv2.putText(img2, tmp_imgfpaths[2].split('_')[-1] + '  ' + metrics[1], (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        img3 = cv2.imread(tmp_imgfpaths[3])
        cv2.putText(img3, tmp_imgfpaths[3].split('_')[-1] + '  ' + metrics[2], (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        img4 = cv2.imread(tmp_imgfpaths[4])
        cv2.putText(img4, tmp_imgfpaths[4].split('_')[-1] + '  ' + metrics[3], (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        img5 = cv2.imread(tmp_imgfpaths[5])
        cv2.putText(img5, tmp_imgfpaths[5].split('_')[-1] + '  ' + metrics[4], (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        img6 = cv2.imread(tmp_imgfpaths[6])
        cv2.putText(img6, tmp_imgfpaths[6].split('_')[-1] + '  ' + metrics[4], (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        img7 = cv2.imread(tmp_imgfpaths[7])
        cv2.putText(img7, tmp_imgfpaths[7].split('_')[-1] + '  ' + metrics[4], (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        img8 = cv2.imread(tmp_imgfpaths[8])
        cv2.putText(img8, tmp_imgfpaths[8].split('_')[-1] + '  ' + metrics[4], (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        img = np.vstack([np.hstack([img0, img1, img2]), np.hstack([img3, img4, img5]), np.hstack([img6, img7, img8])])
        cv2.imshow('viewimg', img)
        key = cv2.waitKey(0)
        if key == ord('q'):
            break
        elif key == ord('j'):
            idx += 500
        elif key == ord('k'):
            idx -= 500
        else:
            # with open('ssim_{}'.format(seq), 'a') as fa:
            #     fa.write(str(ssim) + '\n')
        
            idx += 1
    cv2.destroyAllWindows()
