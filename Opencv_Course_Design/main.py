import cv2 as cv 
import math
import numpy as np
import os 
def Horizontal_correction(path) :
    img = cv.imread(path, cv.IMREAD_COLOR)
    img = cv.resize(img, (700, 700))
    img = img[300 : 650]
    gimg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    canny = cv.Canny(gimg, 70, 150)
    lines = cv.HoughLines(canny, 1, np.pi / 180, 200)
    lines = np.array(lines)
    angles = []

    for line in lines:
        for rho, theta in line :
            a = np.cos(theta)
            b = np.sin(theta)

            x0 = rho * a
            y0 = rho * b

            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * a)
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * a)

#             cv.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

            dy = y2 - y1
            dx = x2 - x1
            angle = math.atan2(dy, dx) * 180 / math.pi
            angles.append(angle)

    angles = sorted(angles, key = lambda x : x)
    best_angle = 0
    angles_temp = []
    i = 0
    while i <= len(angles) - 2 :
        for j in range(i, len(angles)) :
            if abs(angles[j] - angles[i]) < 1e-2 :
                if j == len(angles) - 1 :
                    angles_temp.append(angles[i : j + 1])
                    i = j
                    break
            else :
                angles_temp.append(angles[i : j])
                i = j
                break
    cnt = 0
    for i, lst in enumerate(angles_temp) :
            if cnt < len(lst) :
                cnt = len(lst)
                best_angle = angles_temp[i][0]
        
    RotateMatrix = cv.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), best_angle, 1)
    RotImg = cv.warpAffine(img, RotateMatrix, (img.shape[1], img.shape[0]))

#     concatenate = np.concatenate((img, RotImg), axis = 1)
#     print(RotImg.shape, img.shape)
#     print(angles)
#     cv.imshow('concatenate', concatenate)
#     cv.waitKey(0)
    
    return RotImg

def Rect(contours) :
    areaInitial = 0
    for i in range(len(contours)) :
        contour = contours[i].squeeze()
        min_x, min_y = np.min(contour, axis = 0)
        max_x, max_y = np.max(contour, axis = 0)
#         cv.rectangle(imgGray, (min_x, min_y), (max_x, max_y), color = (0, 0, 255))
        area = (max_x - min_x) * (max_y - min_y)
        if area > areaInitial :
            areaInitial = area
            rectPoint = [min_y, min_x, max_y, max_x]
    return rectPoint

def Biggest(contours) :
    biggest = np.array([])
    max_Area = 0
    for contour in contours :
        area = cv.contourArea(contour)
        if area > 5000:
            C = cv.arcLength(contour, True)
            approx = cv.approxPolyDP(contour, 0.02 * C, True)
            if area > max_Area and len(approx) == 4:
                biggest = approx
                max_Area = area
    return biggest

def FindContours(img) :
    imgGray = img.copy()
    imgBlur = cv.GaussianBlur(imgGray, (5, 5), 1)
    imgCanny = cv.Canny(imgBlur, 100, 200)
    kernel = np.ones((5, 5))
    imgDilate = cv.dilate(imgCanny, kernel, iterations = 2)
    imgErode = cv.erode(imgDilate, kernel, iterations = 1)

    contours, hierarchy = cv.findContours(imgErode, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
#     cv.drawContours(imgGray, contours, -1, (0, 0, 255), 2)
#     cv.imshow('imgGray', imgGray)
#     cv.waitKey(0)
    
    return contours

def Reorder(points):
    points = points.reshape((4, 2))
    Newpoints = np.zeros((4, 2), dtype=np.int32)
    xsum = np.sum(points, axis = 1)
    xdiff = np.diff(points, axis = 1)
    Newpoints[0] = points[np.argmin(xsum)]
    Newpoints[3] = points[np.argmax(xsum)]
    Newpoints[1] = points[np.argmin(xdiff)]
    Newpoints[2] = points[np.argmax(xdiff)]
    
    return np.float32(Newpoints)

def ImageAlignment(img1, img2) :
    contours1  = FindContours(img1)
    contours2 = FindContours(img2)
    
    rectPoint1 = Rect(contours1)
    rectPoint2 = Rect(contours2)
    img1 = img1[rectPoint1[0] : rectPoint1[2], rectPoint1[1] : rectPoint1[3]]
    img2 = img2[rectPoint2[0] : rectPoint2[2], rectPoint2[1] : rectPoint2[3]]
    img2 = cv.resize(img2, (img1.shape[1], img1.shape[0]), cv.INTER_CUBIC)
    
    contours1 = FindContours(img1)
    contours2 = FindContours(img2)
    
    targetAreaPoint1 = Biggest(contours1)
    targetAreaPoint2 = Biggest(contours2)
    pts1 = np.float32(Reorder(targetAreaPoint1))
    pts2 = np.float32(Reorder(targetAreaPoint2))
    pts = np.float32([[0, 0], [img1.shape[1], 0], [0, img1.shape[0]], [img1.shape[1], img1.shape[0]]])
    
    RotateMatrix1 = cv.getPerspectiveTransform(pts1, pts)
    RotateMatrix2 = cv.getPerspectiveTransform(pts2, pts)

    out_img1 = cv.warpPerspective(img1, RotateMatrix1, (img1.shape[1], img1.shape[0]))
    out_img2 = cv.warpPerspective(img2, RotateMatrix2, (img1.shape[1], img1.shape[0]))
#     cv.imshow('out_img1', out_img1)
#     cv.imshow('out_img2', out_img2)
#     concatenate = np.concatenate((out_img1, out_img2), axis = 1)
#     cv.imshow('concatenate', concatenate)
#     cv.imshow('img1', img1)
#     cv.imshow('img2', img2)
#     cv.waitKey(0)
    return out_img1, out_img2

def Detect(img1, img2) :
    cimg1, cimg2 = img1.copy(), img2.copy()
    cimg1 = cv.cvtColor(cimg1, cv.COLOR_BGR2GRAY)
    cimg2 = cv.cvtColor(cimg2, cv.COLOR_BGR2GRAY)
    imgBlur1 = cv.GaussianBlur(cimg1, (3, 3), 0)
    imgBlur2 = cv.GaussianBlur(cimg2, (3, 3), 0)
    kernel = np.ones((3, 3))
    imgErode1 = cv.erode(imgBlur1, kernel, iterations = 1)
    imgErode2 = cv.erode(imgBlur2, kernel, iterations = 1)
#     ret1, imgThresh1 = cv.threshold(imgErode1, 100, 255, cv.THRESH_BINARY)
#     ret2, imgThresh2 = cv.threshold(imgErode2, 100, 255, cv.THRESH_BINARY)
    
    imgCanny1 = cv.Canny(imgErode1, 70, 120)
    imgCanny2 = cv.Canny(imgErode2, 70, 120)
    
    contours1, hierarchy1 = cv.findContours(imgCanny1, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours2, hierarchy2 = cv.findContours(imgCanny2, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    contoursNum = 0
#     cv.drawContours(img1, contours1, -1, (0, 255, 0), 1)
#     cv.drawContours(img2, contours2, -1, (0, 0, 255), 1)
#     concatenate = np.concatenate((img1, img2), axis = 1)
#     cv.imshow('concatenate', concatenate)
    for i in range(len(contours2)) :
        defects = []
        for j in range(len(contours1)) :
            similiarity = cv.matchShapes(contours2[i], contours1[j], cv.CONTOURS_MATCH_I1, 0)
            defects.append(similiarity)
        min_similarity = min(defects)
        if min_similarity > 0.45 :
            cv.drawContours(img2, contours2, i, (0, 255, 0), 1)
            contoursNum += 1 
    
    if contoursNum > 30 :
        print('Approximately defect(s) detected')
    else :
        print('Unable to detect defects')

#     cv.imshow('canny1', img1)
    cv.imshow('canny2', img2)
    cv.waitKey(0)
    
daqipao_path = list(sorted(os.listdir('opencv_course_design_data/NG/daqipao/')))
jiaodai_path = list(sorted(os.listdir('opencv_course_design_data/NG/jiaodai/')))
OK_path = list(sorted(os.listdir('opencv_course_design_data/OK/')))

for path in daqipao_path :
    RotImg1 = RotImg2 = Horizontal_correction('opencv_course_design_data/OK/OK_0032.bmp')
    RotImg2 = Horizontal_correction(os.path.join('opencv_course_design_data/NG/daqipao', path))
    imgAlignment1, imgAlignment2 = ImageAlignment(RotImg1, RotImg2)
    Detect(imgAlignment1, imgAlignment2)
# for path in jiaodai_path :
#     RotImg1 = RotImg2 = Horizontal_correction('opencv_course_design_data/OK/OK_0032.bmp')
#     RotImg2 = Horizontal_correction(os.path.join('opencv_course_design_data/NG/jiaodai', path))
#     imgAlignment1, imgAlignment2 = ImageAlignment(RotImg1, RotImg2)
#     Detect(imgAlignment1, imgAlignment2)
# for path in OK_path :
#     RotImg1 = RotImg2 = Horizontal_correction('opencv_course_design_data/OK/OK_0032.bmp')
#     RotImg2 = Horizontal_correction(os.path.join('opencv_course_design_data/OK/', path))
#     imgAlignment1, imgAlignment2 = ImageAlignment(RotImg1, RotImg2)
#     Detect(imgAlignment1, imgAlignment2)