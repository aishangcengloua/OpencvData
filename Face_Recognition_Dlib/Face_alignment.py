# encoding:utf-8

# import dlib
# import cv2
# import matplotlib.pyplot as plt
# import numpy as np
# import math
#
# def rect_to_bb(rect): # 获得人脸矩形的坐标信息
#     x = rect.left()
#     y = rect.top()
#     w = rect.right() - x
#     h = rect.bottom() - y
#     return (x, y, w, h)
#
# def face_alignment(faces):
#
#     predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") # 用来预测关键点
#     faces_aligned = []
#     for face in faces:
#         rec = dlib.rectangle(0,0,face.shape[0],face.shape[1])
#         print(rec)
#         shape = predictor(np.uint8(face),rec) # 注意输入的必须是uint8类型
#         order = [36,45,30,48,54] # left eye, right eye, nose, left mouth, right mouth  注意关键点的顺序，这个在网上可以找
#         for j in order:
#             x = shape.part(j).x
#             y = shape.part(j).y
#             cv2.circle(face, (x, y), 2, (0, 0, 255), -1)
#
#         eye_center =((shape.part(36).x + shape.part(45).x) * 1./2, # 计算两眼的中心坐标
#                       (shape.part(36).y + shape.part(45).y) * 1./2)
#         dx = (shape.part(45).x - shape.part(36).x) # note: right - right
#         dy = (shape.part(45).y - shape.part(36).y)
#
#         angle = math.atan2(dy,dx) * 180. / math.pi # 计算角度，返回两点与水平线的夹角
#         RotateMatrix = cv2.getRotationMatrix2D(eye_center, angle, scale=1) # 计算仿射矩阵
#         RotImg = cv2.warpAffine(face, RotateMatrix, (face.shape[0], face.shape[1])) # 进行放射变换，即旋转
#         faces_aligned.append(RotImg)
#     return faces_aligned
#
# def demo():
#
#     im_raw = cv2.imread('test.jpg').astype(np.uint8)
#
#     detector = dlib.get_frontal_face_detector()
#     gray = cv2.cvtColor(im_raw, cv2.COLOR_BGR2GRAY)
#     rects = detector(gray, 1)
#
#     src_faces = []
#     for (i, rect) in enumerate(rects):
#         (x, y, w, h) = rect_to_bb(rect)
#         detect_face = im_raw[y:y+h,x:x+w]
#         src_faces.append(detect_face)
#         cv2.rectangle(im_raw, (x, y), (x + w, y + h), (0, 255, 0), 2)
#         cv2.putText(im_raw, "Face: {}".format(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#
#     faces_aligned = face_alignment(src_faces)
#
#     cv2.imshow("src", im_raw)
#     i = 0
#     for face in faces_aligned:
#         cv2.imshow("det_{}".format(i), face)
#         i = i + 1
#     cv2.waitKey(0)
#
# if __name__ == "__main__":
#
#     demo()

import numpy as np
import cv2 as cv
import dlib
import math

def bounding_box(rect) :
    return rect.left(), rect.top(), rect.right(), rect.bottom()

def Faces_alignment(faces) :
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    faces_aligned = []
    for i, face in enumerate(faces) :
        rect = dlib.rectangle(0,0,face.shape[0],face.shape[1])
        print(rect)
        shape = predictor(np.uint8(face), rect)
        order = [36, 45, 30, 48, 54]

        for j in order :
            (x, y) = shape.part(j).x, shape.part(j).y
            cv.circle(face, (x, y), 2, color = (0, 0, 255), thickness = 2, lineType = cv.LINE_AA)
        eye_center = (shape.part(36).x + shape.part(45).x / 2.0,
                      shape.part(36).y + shape.part(45).y / 2.0)

        dx = shape.part(45).x - shape.part(36).x
        dy = shape.part(45).y - shape.part(36).y

        angle = math.atan2(dy, dx) * 180 / math.pi
        RotateMatrix = cv.getRotationMatrix2D(eye_center, angle, 1)
        RotImg = cv.warpAffine(face, RotateMatrix, (face.shape[0], face.shape[1]))

        faces_aligned.append(RotImg)

    return faces_aligned

def demo() :
    image = cv.imread('test.jpg', cv.IMREAD_COLOR)
    detector = dlib.get_frontal_face_detector()
    rects = detector(image, 1)

    faces = []
    for i, rect in enumerate(rects) :
        x1, y1, x2, y2 = bounding_box(rect)
        face = image[y1 : y2, x1 : x2]
        faces.append(face)
        cv.rectangle(image, (x1, y1), (x2, y2), color = (0, 0, 255), thickness = 2, lineType = cv.LINE_AA)
        cv.putText(image, f'Face : {i + 1}', (x1, y1 - 10), fontFace = cv.FONT_HERSHEY_SIMPLEX,
                   fontScale = 0.6, color = (0, 255, 0), thickness = 2, lineType = cv.LINE_AA)

    Faces_aligned = Faces_alignment(faces)
    cv.imshow('image', image)
    for i, face in enumerate(Faces_aligned) :
        cv.imshow(f'det_{i + 1}', face)
    cv.waitKey(0)

if __name__ == '__main__' :
    demo()