import cv2 as cv
import numpy as np

img = cv.imread('images/chess_board.png')
img_gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
img_gaussion = cv.GaussianBlur(img_gray, (5, 5,), 0)
img_canny = cv.Canny(img_gaussion, 70, 120)

lines = cv.HoughLines(img_canny, 1, np.pi / 180, 200)
for line in lines :
    print(line)
    rho, theta = line[0][0], line[0][1]
    a = np.cos(theta)
    b = np.sin(theta)
    x = rho * a
    y = rho * b
    x1, y1 = x + 1000 * b, y - 1000 * a
    x2, y2 = x - 1000 * b, y + 1000 * a
    print(x1, y1)
    cv.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), thickness = 2)
cv.imshow('gray', img)
cv.waitKey(0)

img = cv.imread('images/coins.png')
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img_gray = cv.GaussianBlur(img_gray, (5, 5), 0)
img_gray = cv.medianBlur(img_gray, 7)
cicles = cv.HoughCircles(img_gray, cv.HOUGH_GRADIENT, 1, 180, param1 = 220, param2 = 70).squeeze()
print(cicles.shape)
for cicle in cicles :
    cicle = np.uint16(np.around(cicle))
    print(cicle[0], cicle[1])
    cv.circle(img, center = (cicle[0], cicle[1]), radius = cicle[2], \
              color = (255, 0, 0), thickness = 2)
    cv.circle(img, center = (cicle[0], cicle[1]), radius = 2,
              color = (0, 0, 255), thickness = 3)
cv.imshow('img', img)
cv.waitKey(0)