import cv2
from matplotlib import pyplot as plt
import time
import os
import numpy as np
import torch
from torch.autograd import Function

from cpp_CUDA_code import imgblend_cuda as imgblend

def stitch(path1, path2):
    """
    传入图片路径，返回变换过后的填充图片
    :param path1:
    :param path2:
    :return:
    """
    # 边界填充
    top, bot, left, right = 100, 100, 0, 800
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)
    img1_size = img1.shape[:2]
    img2 = cv2.resize(img2, (img1_size[1], img1_size[0]))

    srcImg = cv2.copyMakeBorder(img1, top, bot, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    testImg = cv2.copyMakeBorder(img2, top, bot, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    img1gray = cv2.cvtColor(srcImg, cv2.COLOR_BGR2GRAY)
    img2gray = cv2.cvtColor(testImg, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d_SIFT().create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1gray, None)
    kp2, des2 = sift.detectAndCompute(img2gray, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Need to draw only good matches, so create a mask
    matchesMask = [[0, 0] for i in range(len(matches))]
    good = []
    pts1 = []
    pts2 = []
    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
            matchesMask[i] = [1, 0]

    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       matchesMask=matchesMask,
                       flags=0)
    img3 = cv2.drawMatchesKnn(img1gray, kp1, img2gray, kp2, matches, None, **draw_params)
    plt.imshow(img3, ), plt.show()

    MIN_MATCH_COUNT = 10
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        warpImg = cv2.warpPerspective(testImg, np.array(M), (testImg.shape[1], testImg.shape[0]),
                                      flags=cv2.WARP_INVERSE_MAP)
        return srcImg, warpImg
    else:
        print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
        matchesMask = None


def blend(srcImg, warpImg, savename=None):
    """
    图片融合，
    """
    rows, cols = srcImg.shape[:2]
    # 找到左右重叠区域
    global left, right
    for col in range(0, cols):
        if srcImg[:, col].any() and warpImg[:, col].any():
            left = col
            break
    for col in range(cols - 1, 0, -1):
        if srcImg[:, col].any() and warpImg[:, col].any():
            right = col
            break
    print(left, right)

    width, height = srcImg.shape[:2]
    print(srcImg.shape)

    cv2.imwrite('../data/src.jpg', srcImg)
    cv2.imwrite('../data/warp.jpg', warpImg)

    leftimg = torch.from_numpy(srcImg).int().cuda()
    rightimg = torch.from_numpy(warpImg).int().cuda()
    res = torch.ones([width, height, 3]).int().cuda()

    t1 = time.time()
    imgblend.imgblend_wrapper(width, height, left, right, leftimg, rightimg, res)
    print(time.time() - t1)

    img = res.cpu().numpy()
    img = np.array(img, dtype=np.uint8)

    # res = np.zeros([rows, cols, 3], np.uint8)
    # alpha = np.zeros((rows, right - left, 3), dtype=np.float)
    # for row in range(0, rows):
    #     for col in range(left, right):
    #         if not srcImg[row, col].any():  # src不存在
    #             alpha[row, col - left, :] = 0
    #         elif not warpImg[row, col].any():  # warpImg 不存在
    #             alpha[row, col - left, :] = 1
    #         else:  # src 和warp都存在
    #             srcImgLen = float(abs(col - left))
    #             testImgLen = float(abs(col - right))
    #             alpha[row, col - left, :] = testImgLen / (srcImgLen + testImgLen)
    #
    # res[:, :left] = srcImg[:, :left]
    # res[:, right:] = warpImg[:, right:]
    # res[:, left:right] = np.clip(srcImg[:, left:right] * alpha + warpImg[:, left:right] * (np.ones_like(alpha) - alpha),
    #                              0, 255)
    #
    # # opencv is bgr, matplotlib is rgb
    # res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
    # if savename is not None:
    #     plt.imsave(savename, res)
    return img


if __name__ == "__main__":
    # path1 = 'data/1.jpg'
    # path2 = 'data/2.jpg'
    # path3 = 'data/3.jpg'
    # path4 = 'data/4.jpg'
    # path5 = 'stitch12.jpg'
    # path6 = 'stitch34.jpg'
    # res12 = 'data/stitch12.jpg'
    # res34 = 'data/stitch34.jpg'

    path1 = '../data/1.jpg'
    path2 = '../data/2.jpg'
    srcImg, warpImg = stitch(path1, path2)
    cv2.imshow('src', srcImg)
    cv2.imshow('warp', warpImg)
    cv2.waitKey(1000)
    # cv2.imwrite('srcImg.jpg', srcImg)
    # cv2.imwrite('warpImg.jpg', warpImg)
    start = time.time()
    res = blend(srcImg, warpImg)
    end = time.time()
    print('融合时间：', end - start)
    # show the result
    res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
    plt.figure()
    plt.imshow(res)
    plt.show()

