import cv2 as cv
import numpy as np
import dlib

def bounding_box(rect) :
    x1 = rect.left()
    y1 = rect.top()
    x2 = rect.right()
    y2 = rect.bottom()

    return x1, y1, x2, y2

def resize_image(image) :
    image = cv.resize(image, (1000, 800))
    return image

def detect_faces() :
    image = cv.imread('test.jpg', cv.IMREAD_COLOR)
    image = resize_image(image)
    detector = dlib.get_frontal_face_detector()
    rects = detector(image, 1)

    for i, rect in enumerate(rects) :
        x1, y1, x2, y2 = bounding_box(rect)
        cv.rectangle(image, (x1, y1), (x2, y2), color = (0, 255, 0), thickness = 2, lineType = cv.LINE_AA)
        cv.putText(image, f"Face: {i + 1}", org = (x1, y1 - 10), fontFace = cv.FONT_HERSHEY_SIMPLEX,
                   fontScale = 0.6, color = (0, 0, 255), thickness = 2)
    cv.imshow('img', image)
    cv.waitKey(0)

if __name__ == '__main__' :
    detect_faces()