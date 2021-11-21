import cv2
import os
import copy
import math
import numpy as np
import math
import matplotlib.pyplot as plt


def style_transfer(image, ref):
    out = np.zeros_like(ref)
    hist_img, _ = np.histogram(image[:, :], 256)
    hist_ref, _ = np.histogram(ref[:, :], 256)
    cdf_img = np.cumsum(hist_img)
    cdf_ref = np.cumsum(hist_ref)

    for j in range(256):
        tmp = abs(cdf_img[j] - cdf_ref)
        tmp = tmp.tolist()
        idx = tmp.index(min(tmp))  # 找出tmp中最小的数，得到这个数的索引
        out[image[:, :] == j] = idx
    return out

img1 = cv2.imread('./images/shenzhen_gray.bmp', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('./images/dark-city.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
out = style_transfer(img1, img2)

cv2.imshow("img1", img1)
cv2.imshow("img2", img2)
cv2.imshow("out", out)
cv2.waitKey(-1)
cv2.destroyAllWindows()

####################### Filtering

def median_filter(img, filter_size):
    newImg = copy.deepcopy(img)
    offset = int((filter_size - 1) / 2)
    for r in range(offset, img.shape[0]-offset):
        for c in range(offset, img.shape[1]-offset):
            blk = img[r-offset:r+offset+1, c-offset:c+offset+1]
            newImg[r, c] = np.median(blk)
    return newImg

img_clean = cv2.imread('./images/shenzhen_gray.bmp', cv2.IMREAD_GRAYSCALE)
img_noise = np.array(img_clean)
for k in range(10000):
    xj = int(np.random.uniform(0, img2.shape[0]))
    xi = int(np.random.uniform(0, img2.shape[1]))
    img_noise[xj, xi] = 255

result_mean = cv2.blur(img_noise, (3,3))
cv2.imshow("noise image", img_noise)
cv2.imshow("mean filter", result_mean)

result_median = cv2.medianBlur(img_noise, 3)
cv2.imshow("median filter", result_median)

result_median2 = median_filter(img_noise, 3)
cv2.imshow("my median filter", result_median2)
cv2.waitKey(-1)
cv2.destroyAllWindows()


####################### EDGE Detection #################################
img_clean = cv2.imread('./images/writing.jpg', cv2.IMREAD_GRAYSCALE)
img_clean = cv2.resize(img_clean, (700, 500))
cv2.imshow("original", img_clean)

def filter_kernel(img, kernel):
    newImg = copy.deepcopy(img)
    filter_size = kernel.shape[0]
    offset = int((filter_size - 1) / 2)
    for r in range(offset, img.shape[0]-offset):
        for c in range(offset, img.shape[1]-offset):
            blk = img[r-offset:r+offset+1, c-offset:c+offset+1]
            newImg[r,c] = abs(np.sum(np.multiply(kernel, blk)))
    return newImg

result_sobel_x = cv2.Sobel(img_clean, -1, 1,0, ksize=3)
result_sobel_y = cv2.Sobel(img_clean, -1, 0,1, ksize=3)
cv2.imshow("Sobel X", result_sobel_x)
cv2.imshow("Sobel y", result_sobel_y)

# Laplacian filtering
result_laplace0 = cv2.Laplacian(img_clean, -1, ksize=3)
cv2.imshow("Laplace Filter 0", result_laplace0)

laplacien1 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
result_laplace1 = filter_kernel(img_clean, laplacien1)
cv2.imshow("Laplace Filter 1", result_laplace1)

laplacien2 = np.array(([-1,-1,-1],[-1,8,-1],[-1,-1,-1]))
result_laplace2 = filter_kernel(img_clean, laplacien2)
cv2.imshow("Laplace Filter 2", result_laplace2)

cv2.waitKey(-1)




