import numpy as np
import cv2 as cv
import dlib

def bounding_box(rect) :
    x1, y1 = rect.left(), rect.top()
    x2, y2 = rect.right(), rect.bottom()

    return x1, y1, x2, y2

def reize_image(image, proportion) :
    image = cv.resize(image, (int(image.shape[1] * proportion), int(image.shape[0] * proportion)))
    return image

def shape_to_np(shape) :
    point = np.zeros((68, 2), dtype = int)
    for idx, pt in enumerate(shape.parts()) :
        point[idx] = (pt.x, pt.y)

    return point

def feature_detection() :
    image = cv.imread('test.jpg', cv.IMREAD_COLOR)
    image = reize_image(image, 0.4)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    rects = detector(image, 1)
    shapes = []
    for i, rect in enumerate(rects) :
        shape = predictor(image, rect)
        shape = shape_to_np(shape)
        shapes.append(shape)

        x1, y1, x2, y2 = bounding_box(rect)
        cv.rectangle(image, (x1, y1), (x2, y2), lineType = cv.LINE_AA, thickness = 2, color = (0, 0, 255))
        cv.putText(image, f'Face : {i + 1}', (x1, y1 - 10), fontFace = cv.FONT_HERSHEY_SIMPLEX,
                   fontScale = 0.6, lineType = cv.LINE_AA, color = (0, 255, 0), thickness = 2)

    for shape in shapes:
        for x, y in shape :
            cv.circle(image, (x, y), 2, color = (0, 0, 255), lineType = cv.LINE_AA)

    cv.imshow('image', image)
    cv.waitKey(0)

if __name__ == '__main__' :
    feature_detection()