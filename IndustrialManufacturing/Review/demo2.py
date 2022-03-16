import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('peach.jpg', 1)
#因为cv2读取的照片类型是BGR类型，所以要转成RGB类型的照片
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#得到r, g, b通道的照片
r, g, b = cv2.split(img_rgb)
#获得RG灰度图像
c = r - g
#求出色差图的直方图，查看分割的最优阈值
hist, bins = np.histogram(c, bins = 256, range = (0, 256))
plt.plot(hist)
plt.show()
#采用190作为阈值
thresh_value = np.sum(c[np.where(c != 0)]) / np.sum(c != 0)
_, peach = cv2.threshold(c, 190, 255, cv2.THRESH_BINARY_INV)
#进行腐蚀操作，将小白点去除
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
peach = cv2.erode(peach, kernel, iterations = 3)


cv2.imshow('peach', peach)
cv2.waitKey(0)

# wheat1 = cv2.imread('wheat1.jpg', 1)
# wheat2 = cv2.imread('wheat2.jpeg', 1)
# wheats = [wheat1, wheat2]
#
# for wheat in wheats :
#     wheat_rgb = cv2.cvtColor(wheat, cv2.COLOR_BGR2RGB)
#     r, g, b = cv2.split(wheat_rgb)
#     c = 2 * g - r - b
#     thresh_value = np.mean(c)
#     _, wheatSeeding = cv2.threshold(c, thresh_value, 255, cv2.THRESH_OTSU)
#     cv2.imshow('wheatSeeding', wheatSeeding)
#     cv2.waitKey(0)