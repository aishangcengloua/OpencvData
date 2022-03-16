import numpy as np
import cv2 as cv
def circle_flatten() :
    img = cv.imread('20181016134344221.jpg')
    print(img.shape)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # img_gray = cv.medianBlur(img_gray, 3)
    print(img_gray.shape)

    circles = cv.HoughCircles(img_gray, cv.HOUGH_GRADIENT, 1, 50, param1 = 150, param2 = 50).squeeze(0)
    print(circles)
    for circle in circles :
        circle = np.uint16(np.around(circle))
        cv.circle(img, (circle[0], circle[1]), radius=circle[2], color=(0, 0, 255), thickness=5)
        cv.circle(img, (circle[0], circle[1]), radius=2, color=(255, 0, 0), thickness=2)

    #获得检测到的所有圆的半径
    circle_radius = circles[ : , 2]
    print(circle_radius)
    #获得最大半径的下标
    radius_biggest_index = np.argsort(circle_radius)[-1]
    print(radius_biggest_index)
    # #做出最大圆
    circle = np.uint16(np.around(circles[radius_biggest_index]))
    # cv.circle(img, (circle[0], circle[1]), radius = circle[2], color = (0, 0, 255), thickness = 5)
    # cv.circle(img, (circle[0], circle[1]), radius = 2, color = (255, 0, 0), thickness = 2)

    #取展平后条形圆环的宽为最大半径的一半，而长取最大圆的周长
    height = int(circle_radius[radius_biggest_index] * np.pi * 2)
    # print(height)
    width = int(circle_radius[radius_biggest_index] / 2)
    rectangle = np.zeros([width, height])
    cv.imshow('2', rectangle)
    print(height, width)
    print(rectangle.shape)
    count = 0
    for row in range(width) :
        for col in range(height) :
            theta = np.pi * 2.0 / height * (col + 1)
            rho = circle_radius[radius_biggest_index] - row
            position_x = int(circle[0] + rho * np.cos(theta) + 0.5)
            position_y = int(circle[1] - rho * np.sin(theta) + 0.5)
            rectangle[row][col] = (img_gray[position_y, position_x])
            if img_gray[position_y, position_x] == 255 :
                count += 1
    rectangle = np.uint8(rectangle)
    print(np.sum((rectangle == 255)))
    print(count)
    print(rectangle)
    cv.imshow('1', rectangle)
    cv.imshow('img', img)
    cv.waitKey(0)

circle_flatten()