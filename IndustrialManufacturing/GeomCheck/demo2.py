import cv2 as cv
import numpy as np

img = cv.imread('images/shenzhen.png')
src_h, src_w, src_c= img.shape
dst_h, dst_w, dst_c = int(src_h / 2), int(src_w / 2), src_c

dst_img = np.zeros((dst_h, dst_w, dst_c), dtype = np.uint8)
for h in range(dst_img.shape[0]) :
    for w in range(dst_img.shape[1]) :
        New_h = int(h * src_h * 1.0 / dst_h)
        New_w = int(w * src_w * 1.0 / dst_w)
        dst_img[h][w] = img[New_h][New_w]

Rotated_matrix = cv.getRotationMatrix2D(center = (int(src_w / 2), int(src_h / 2)), angle = 15, scale = 1)
Rotated_img = cv.warpAffine(img, Rotated_matrix, (img.shape[1], img.shape[0]))
cv.imshow('rotated_img', Rotated_img)
cv.imshow('src_img', img)
cv.imshow('img', dst_img)
cv.waitKey(0)