import cv2 as cv
import numpy as np
import copy

#3.1
def line() :
    blank = np.full((300, 300, 3), 255, dtype = np.uint8)
    # cv.line(blank, (150 ,150), (250, 150), (0, 0, 255), thickness = 2)
    #获得旋转矩阵
    RotatedMatirx = cv.getRotationMatrix2D((20, 20), -60, 1)
    pts = np.array([[150, 150, 1], [250, 150, 1]])
    #获得旋转后对应的点
    ptsRotated = np.multiply(pts, RotatedMatirx)
    cv.line(blank, (int(ptsRotated[0, 0]), int(ptsRotated[0, 1])),
            (int(ptsRotated[1, 0]), int(ptsRotated[1, 1])),
            (0, 0, 255), thickness = 2)
    cv.circle(blank, (20, 20), radius = 2,
              color = (0, 255, 0), thickness = 3)
    cv.putText(blank, f'center: (20, 20)', (10, 40),
               fontFace = cv.FONT_HERSHEY_SIMPLEX,
               fontScale = 0.6, color = (255, 255, 0), thickness = 2)
    cv.imshow('balnk', blank)
    cv.waitKey(0)

#3.2
def circle_detect() :
    img = cv.imread('images/weiqi.png')
    cimg = copy.deepcopy(img)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # img_gray = cv.GaussianBlur(img_gray, (5, 5), 0)
    #降噪
    img_gray = cv.medianBlur(img_gray, 5)
    circles = cv.HoughCircles(img_gray, cv.HOUGH_GRADIENT, 1, 10, param1 = 120, param2 = 50).squeeze()

    for circle in circles :
        #转成np.uint16数据类型扩大数据范围
        circle = np.uint16(np.around(circle))
        #做二值化处理，白棋为白，黑棋为黑，若圆心的值为255则为白棋，否则为黑棋
        _, img_thresh = cv.threshold(img_gray, 80, 255, cv.THRESH_BINARY)
        #注意x和y与w和h的对应
        if img_thresh[circle[1], circle[0]] == 255 :
            print(f'chess is black!')
            # 画出检测到的圆
            cv.circle(img, center=(circle[0], circle[1]), radius=circle[2],
                      color=(255, 0, 0), thickness=2)
            # 画出圆心
            cv.circle(img, center=(circle[0], circle[1]), radius=2,
                      color=(255, 0, 0), thickness=3)
        else :
            print('chess is white!')
            # 画出检测到的圆
            cv.circle(img, center=(circle[0], circle[1]), radius=circle[2],
                      color=(0, 0, 255), thickness=2)
            # 画出圆心
            cv.circle(img, center=(circle[0], circle[1]), radius=2,
                      color=(0, 0, 255), thickness=3)
        cv.imshow('canny', img_thresh)
    # img_concatenate = np.concatenate((cimg, img), axis = 1)
    # cv.imwrite('img_con.png', img_concatenate)
    cv.imshow('img', img)
    cv.waitKey(0)

#3.3
def line_detect() :
    img = cv.imread('images/chemical_tube.png')
    img_gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    img_gaussion = cv.GaussianBlur(img_gray, (5, 5,), 0)
    img_canny = cv.Canny(img_gaussion, 70, 120)

    x1_, x2_, y1_, y2_= [], [], [], []
    lines = cv.HoughLines(img_canny, 1, np.pi / 180, 300)
    for line in lines:
        rho, theta = line[0][0], line[0][1]
        a = np.cos(theta)
        b = np.sin(theta)
        x = rho * a
        y = rho * b
        x1, y1 = x + 1000 * b, y - 1000 * a
        x2, y2 = x - 1000 * b, y + 1000 * a
        cv.line(img, (int(x1), int(y1)), (int(x2), int(y2)),
                (255, 0, 0), thickness=2)
        #收集每条直线的两个点
        x1_.append(x1)
        x2_.append(x2)
        y1_.append(y1)
        y2_.append(y2)
    #因为目标的三条直线自上而下的第二到第四条，也就是y在第二到第四的位置
    y_arr = np.array(y1_)
    #获得下标
    y_target_index = np.argsort(y_arr)[1 : 4].tolist()
    k = []
    #计算三条直线的斜率
    for index in y_target_index :
        k.append((y2_[index] - y1_[index]) / (x2_[index] - x1_[index]))
    # print(k)
    #当他们的斜率之间差值小于1e-3时，则认为他们相互平行
    if abs(k[0] - k[1]) < 1e-3 and abs(k[0] - k[2]) < 1e-3 and abs(k[1] - k[2]) < 1e-3:
        print(f'The three lines are parallel\nTheir slopes are:{k}')

    cv.imshow('gray', img)
    cv.waitKey(0)

#3.4
def circle_flatten() :
    img = cv.imread('images/circle_band.bmp')
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # img_gray = cv.medianBlur(img_gray, 3)

    circles = cv.HoughCircles(img_gray, cv.HOUGH_GRADIENT, 1, 50, param1 = 170, param2 = 100).squeeze()
    #获得检测到的所有圆的半径
    circle_radius = circles[ : , 2]
    #获得最大半径的下标
    radius_biggest_index = np.argsort(circle_radius)[-1]
    print(radius_biggest_index)
    #做出最大圆
    circle = np.uint16(np.around(circles[radius_biggest_index]))
    cv.circle(img, (circle[0], circle[1]), radius = circle[2], color = (0, 0, 255), thickness = 5)
    cv.circle(img, (circle[0], circle[1]), radius = 2, color = (255, 0, 0), thickness = 2)

    #取展平后条形圆环的宽为最大半径的一半，而长取最大圆的周长
    height = int(circle_radius[radius_biggest_index] * np.pi * 2)
    width = int(circle_radius[radius_biggest_index] / 3)
    rectangle = np.zeros([width, height])
    print(rectangle.shape)
    print(img_gray.shape)
    for row in range(width) :
        for col in range(height) :
            #转成极坐标系
            theta = np.pi * 2.0 / height * (col + 1)
            rho = circle_radius[radius_biggest_index] - row - 1
            #以圆心为原点，求得原来圆环对应的坐标
            position_x = int(circle[0] + rho * np.cos(theta) + 0.5)
            position_y = int(circle[1] - rho * np.sin(theta) + 0.5)
            rectangle[row, col] = img_gray[position_y, position_x]
    #要转回np.uint8型数据，否则显示有问题
    rectangle = np.uint8(rectangle)
    cv.imwrite('flatten.png', rectangle)
    cv.imshow('1', rectangle)
    cv.imshow('img', img)
    cv.waitKey(0)

if __name__ == '__main__':
    # line()
    # circle_detect()
    # line_detect()
    circle_flatten()