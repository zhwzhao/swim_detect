import os
import time
import cv2
import numpy as np
import torch
from torch.autograd import Function

from cpp_CUDA_code import imgblend_cuda as imgblend

if __name__ == '__main__':
    left = 313  # 400  # 300
    right = 1079  # 399  # 399

    leftimg = cv2.imread('../data/src.jpg')
    rightimg = cv2.imread('../data/warp.jpg')
    width, height = leftimg.shape[:2]

    leftimg = torch.from_numpy(leftimg).int().cuda()
    rightimg = torch.from_numpy(rightimg).int().cuda()
    # print(leftimg.shape)
    # leftimg = torch.rand([width, height, 3]) * 255  # torch.randint(1, 255, (width, height, 3))
    # rightimg = torch.rand([width, height, 3]) * 255  # torch.randint(1, 255, (width, height, 3))
    idx = torch.ones([width, height, 3]).int().cuda()

    t1 = time.time()
    imgblend.imgblend_wrapper(width, height, left, right, leftimg, rightimg, idx)
    print(time.time() - t1)

    img = idx.cpu().numpy()
    img = np.array(img, dtype=np.uint8)
    print(img.shape)
    cv2.imshow('img', img)
    cv2.waitKey(0)
