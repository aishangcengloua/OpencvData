import cv2
import os
import copy
import math
import numpy as np
import math
import random
import matplotlib.pyplot as plt

def test_histogram():
    cImg = cv2.imread('./images/shenzhen.png', cv2.IMREAD_COLOR)
    gImg = cv2.cvtColor(cImg, cv2.COLOR_BGR2GRAY)
    gImg_equ = cv2.equalizeHist(gImg)
    cv2.imshow('original', gImg)
    cv2.imshow('gImg_equ', gImg_equ)
    cv2.waitKey(-1)

def test_spatial_filtering():
    img_clean = cv2.imread('./images/shenzhen_gray.bmp', cv2.IMREAD_GRAYSCALE)
    img_noise = np.array(img_clean)
    for k in range(10000):
        xj = int(np.random.uniform(0, img_clean.shape[0]))
        xi = int(np.random.uniform(0, img_clean.shape[1]))
        img_noise[xj, xi] = 255

    result_mean = cv2.blur(img_noise, (3, 3))
    cv2.imshow("noise image", img_noise)
    cv2.imshow("mean filter", result_mean)

    result_median = cv2.medianBlur(img_noise, 3)
    cv2.imshow("median filter", result_median)

    result_gaussian = cv2.GaussianBlur(img_noise, (3,3), 0)
    cv2.imshow("gaussian filter", result_gaussian)

    cv2.waitKey(-1)

def test_edge_detection():
    img_clean = cv2.imread('./images/writing.jpg', cv2.IMREAD_GRAYSCALE)
    img_clean = cv2.resize(img_clean, (700, 500))
    cv2.imshow("original", img_clean)

    result_sobel_x = cv2.Sobel(img_clean, -1, 1, 0, ksize=3)
    result_sobel_y = cv2.Sobel(img_clean, -1, 0, 1, ksize=3)
    cv2.imshow("Sobel X", result_sobel_x)
    cv2.imshow("Sobel y", result_sobel_y)

    combined_soble = cv2.addWeighted(result_sobel_x, 0.5, result_sobel_y, 0.5, 0)
    cv2.imshow('combined_soble', combined_soble)


    ### LOG
    # 先通过高斯滤波降噪
    gaussian = cv2.GaussianBlur(img_clean, (3, 3), 0)
    dst = cv2.Laplacian(gaussian, cv2.CV_16S, ksize=3)
    log_result = cv2.convertScaleAbs(dst)
    cv2.imshow("LOG", log_result)

    canny = cv2.Canny(img_clean, 80, 200)
    cv2.imshow("canny", canny)

    cv2.waitKey(-1)

def test_binary_segmentation():
    img_clean = cv2.imread('./images/shenzhen_gray.bmp', cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img_clean, (700, 500))
    # cv2.imshow("original", img)

    # 应用5种不同的阈值方法
    ret, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    ret, th2 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    ret, th3 = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
    ret, th4 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
    ret, th5 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)

    titles = ['Original', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
    images = [img, th1, th2, th3, th4, th5]
    plt.figure('Fixed Threshold Binarization')
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.imshow(images[i], 'gray')
        plt.title(titles[i], fontsize=8)
        plt.xticks([]), plt.yticks([])  # 隐藏坐标轴

    ret2,th6 = cv2.threshold(img,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    titles = ['OTSU']
    images = [img, th6]
    plt.figure('OTSU')
    plt.imshow(th6,'gray')
    plt.show()

def test_morph():
    img1 = cv2.imread('./images/morph_kai.png', cv2.IMREAD_GRAYSCALE)
    cv2.imshow('original 1', img1)

    kernel = np.ones((15, 15), np.uint8)
    result = cv2.morphologyEx(img1, cv2.MORPH_OPEN, kernel)
    cv2.imshow('MORPH_OPEN', result)

    img2 = cv2.imread('./images/morph_bi.png', cv2.IMREAD_GRAYSCALE)
    cv2.imshow('original 2', img2)

    kernel = np.ones((13, 13), np.uint8)
    result = cv2.morphologyEx(img2, cv2.MORPH_CLOSE, kernel)
    cv2.imshow('MORPH_CLOSE', result)

def test_contour():
    img = cv2.imread('./images/star.png', cv2.IMREAD_GRAYSCALE)
    x, img = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)

    cv2.imshow("img", img)
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    print(len(contours))

    img2 = np.zeros((img.shape[0], img.shape[1], 3))
    cv2.drawContours(img2, contours, -1, (0, 0, 255), 3)
    cv2.imshow("img2", img2)

    img3 = np.zeros((img.shape[0], img.shape[1]))
    for k in range(contours[0].shape[0]):
        pt = contours[0][k][0]
        img3[pt[1], pt[0]] = 255
        cv2.imshow("img3", img3)
        cv2.waitKey(1)
    cv2.waitKey(-1)

def test_harris():
    filename = './images/chess_board.png'
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    cv2.imshow('original', gray)

    #找到Harris角点
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,2,3,0.04)

    dst = cv2.dilate(dst,None)
    ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
    dst = np.uint8(dst)
    cv2.imshow('harris', dst)

    #找到重心
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

    #定义迭代次数
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)

    # 绘制角点
    centroids = np.int0(centroids)
    corners = np.int0(corners)
    for pt in corners:
        cv2.circle(img, pt, 1, (0,0,255), 2)
    for pt in centroids:
        cv2.circle(img, pt, 1, (0,255,0), 2)

    cv2.imshow('corners', img)
    cv2.waitKey(-1)

def test_orb_rotation():
    img1 = cv2.imread('./images/usb1.jpg', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('./images/usb3.jpg', cv2.IMREAD_GRAYSCALE)
    cv2.imshow('img1', img1)
    cv2.imshow('img2', img2)

    orb = cv2.ORB_create(500)

    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # bf匹配
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des1, trainDescriptors=des2, k=2)
    good = [m for (m, n) in matches if m.distance < 0.35 * n.distance]

    min_match_count = 5
    if len(good) > min_match_count:
        # 匹配点
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        # 找到变换矩阵m
        m, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5, 0)
        matchmask = mask.ravel().tolist()
        h, w = img1.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, m)
        cv2.polylines(img2, [np.int32(dst)], True, 0, 10, cv2.LINE_AA)

        # warp image1 toward image2
        im_out = cv2.warpPerspective(img1, m, img1.shape)
        cv2.imshow("img1 rotated", im_out)
    else:
        print('not enough matches are found  -- %d/%d'(len(good), min_match_count))
        matchmask = None

    draw_params = dict(matchColor=(0, 255, 0), singlePointColor=None,
                       matchesMask=matchmask, flags=2)
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
    cv2.imshow("img3", img3)
    cv2.waitKey(-1)

def test_shape_match():
    # 载入原图
    img = cv2.imread('./images/abc.jpg', 0)
    # 在下面这张图像上作画
    image1 = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

    # 二值化图像
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 搜索轮廓
    contours, hierarchy = cv2.findContours(thresh, 3, 2)
    hierarchy = np.squeeze(hierarchy)

    # 载入标准模板图
    img_a = cv2.imread('./images/A2.jpg', 0)
    _, th = cv2.threshold(img_a, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours1, hierarchy1 = cv2.findContours(th, 3, 2)
    # 字母A的轮廓
    template_a = contours1[0]

    # 记录最匹配的值的大小和位置
    min_pos = -1
    min_value = 2
    for i in range(len(contours)):
        # 参数3：匹配方法；参数4：opencv预留参数
        value = cv2.matchShapes(template_a,contours[i],1,0.0)
        if value < min_value:
            min_value = value
            min_pos = i

    # 参数3为0表示绘制本条轮廓contours[min_pos]
    cv2.drawContours(image1,[contours[min_pos]],0,[255,0,0],3)

    cv2.imshow('result',image1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    test_harris()