import cv2
import numpy as np
import cv2 as cv
from numpy.linalg import inv

def _invertAffineTransform(matrix):
    """cv.invertAffineTransform(). 本质是求逆
    :param matrix: shape[2, 3]. float32
    :return: shape[2, 3]
    """
    matrix = np.concatenate([matrix, np.array([0, 0, 1], dtype=matrix.dtype)[None]])  # for求逆
    return inv(matrix)[:2]

def MyWarpAffine(x, matrix, dsize=None, flags=None):
    """cv.warpAffine(borderMode=None, borderValue=(114, 114, 114))

    :param x: shape[H, W, C]. uint8
    :param matrix: 仿射矩阵. shape[2, 3]. float32
    :param dsize: Tuple[W, H]. 输出的size
    :param flags: cv.WARP_INVERSE_MAP. 唯一可选参数
    :return: shape[dsize[1], dsize[0], C]. uint8
    """
    dsize = dsize or (x.shape[1], x.shape[0])  # 输出的size
    borderValue = np.array((114, 114, 114), dtype=x.dtype)  # 背景填充
    if flags is None or flags & cv.WARP_INVERSE_MAP == 0:  # flags无cv.WARP_INVERSE_MAP参数
        matrix = _invertAffineTransform(matrix)
    grid_x, grid_y = np.meshgrid(np.arange(dsize[0]), np.arange(dsize[1]))  # np.int32
    src_x = (matrix[0, 0] * grid_x + matrix[0, 1] * grid_y + matrix[0, 2]).round().astype(np.int32)  # X
    src_y = (matrix[1, 0] * grid_x + matrix[1, 1] * grid_y + matrix[1, 2]).round().astype(np.int32)  # Y
    src_x_clip = np.clip(src_x, 0, x.shape[1] - 1)  # for索引合法
    src_y_clip = np.clip(src_y, 0, x.shape[0] - 1)
    output = np.where(((0 <= src_x) & (src_x < x.shape[1]) & (0 <= src_y) & (src_y < x.shape[0]))[:, :, None],
                      x[src_y_clip, src_x_clip], borderValue[None, None])  # 广播机制
    return output


class BBox(object):
    def __init__(self, bbox):
        self.left = bbox[0]
        self.top = bbox[1]
        self.right = bbox[2]
        self.bottom = bbox[3]

def test_affine() :
    img = cv2.imread('./images/shenzhen.png', cv2.IMREAD_COLOR)

    h, w, c = img.shape
    box = [0, 0, w, h]
    bbox = BBox(box)

    center = ((bbox.left + bbox.right) / 2, (bbox.top + bbox.bottom) / 2)
    rot_mat = cv2.getRotationMatrix2D(center, 15, 1)
    print(rot_mat)

    img_rotated = cv2.warpAffine(img, rot_mat, (img.shape[1], img.shape[0]))

    img_rotated2 = MyWarpAffine(img, rot_mat, (img.shape[1], img.shape[0]))
    cv2.imshow('my Affine', img_rotated2)

    # 获得图片旋转以后的关键点的位置
    offset = 20
    landmarks = np.array([
        (center[0] - offset, center[1] - offset),
        (center[0] - offset, center[1]),
        center]).astype(np.int)
    print("landmarks:")
    print(landmarks)
    for k in range(1, len(landmarks)):
        a = tuple(landmarks[k - 1])
        b = tuple(landmarks[k])
        cv2.line(img, a, b, (0, 0, 255), 2)
    cv2.line(img, landmarks[2], landmarks[0], (0, 0, 255), 2)
    cv2.imshow('img', img)

    # for points
    src = np.zeros((len(landmarks), 1, 2))
    src[:, 0] = landmarks
    landmarks2 = cv2.transform(src, rot_mat).squeeze()
    landmarks2 = landmarks2.astype(np.int)
    print("landmarks2:")
    print(landmarks2)

    # my transform
    landmarks3 = []
    for k in range(len(landmarks)):
        x, y = landmarks[k]
        d_x = (rot_mat[0, 0] * x + rot_mat[0, 1] * y + rot_mat[0, 2]).round().astype(np.int32)  # X
        d_y = (rot_mat[1, 0] * x + rot_mat[1, 1] * y + rot_mat[1, 2]).round().astype(np.int32)  # Y
        landmarks3.append((d_x, d_y))
    print("landmarks3:")
    print(landmarks3)

    keypts = landmarks3
    for k in range(1, len(keypts)):
        a = tuple(keypts[k - 1])
        b = tuple(keypts[k])
        cv2.line(img_rotated, a, b, (0, 0, 255), 2)
    cv2.line(img_rotated, keypts[2], keypts[0], (0, 0, 255), 2)
    cv2.imshow('img_rotated', img_rotated)

    cv2.waitKey(-1)

def test_polar_coordinate():
    image = 255 * np.ones((255,255,3), np.uint8)
    cv2.imshow("image", image)

    r = 80
    theta = 30 * np.pi / 180
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*r
    y0 = b*r
    x1 = int(x0 + 100*(-b))
    y1 = int(y0 + 100*(a))
    x2 = int(x0 - 100*(-b))
    y2 = int(y0 - 100*(a))
    print([x1,y1,x2,y2])
    cv2.line(image,(x1,y1), (x2,y2), (0,0,255),2)
    cv2.imshow("image2", image)
    cv2.waitKey(-1)

def test_hough_line():
    # image = cv2.imread('./images/lanes.jpg', cv2.IMREAD_COLOR)
    # gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # blurred_image = cv2.GaussianBlur(gray_image, (9, 9), 0)
    # edges_image = cv2.Canny(blurred_image, 50, 120)
    # edges_image[0:120,:] = 0
    # cv2.imshow("edges_image", edges_image)
    # rho_resolution = 1
    # theta_resolution = np.pi/180
    # threshold = 120
    # lines = cv2.HoughLines(edges_image, rho_resolution , theta_resolution , threshold).squeeze()

    image = cv2.imread('./images/buildings.jpg', cv2.IMREAD_COLOR)
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (9, 9), 0)
    edges_image = cv2.Canny(blurred_image, 50, 120)
    cv2.imshow("edges_image", edges_image)
    rho_resolution = 1
    theta_resolution = np.pi/180
    threshold = 155
    lines = cv2.HoughLines(edges_image, rho_resolution , theta_resolution , threshold).squeeze()

    if lines is not None:
        print(len(lines))
        for params in lines:
            r = params[0]
            theta = params[1]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*r
            y0 = b*r
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            cv2.line(image,(x1,y1), (x2,y2), (0,0,255),2)
        cv2.imshow('image', image)
        cv2.waitKey(-1)
    else:
        print("Failed")


def test_hough_circle():
    coins_img = cv2.imread('./images/coins.png')
    coins_img_gray = cv2.cvtColor(coins_img, cv2.COLOR_BGR2GRAY)

    binary = cv2.adaptiveThreshold(coins_img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    coins_circle = cv2.HoughCircles(binary,
                                    cv2.HOUGH_GRADIENT,
                                    2,
                                    100,
                                    param1=350,
                                    param2=260,
                                    minRadius=100,
                                    maxRadius=180)

    circles = coins_circle.reshape(-1, 3)
    circles = np.uint16(np.around(circles))

    for i in circles:
        cv2.circle(coins_img, (i[0], i[1]), i[2], (0, 0, 255), 5)  # 画圆
        cv2.circle(coins_img, (i[0], i[1]), 2, (0, 255, 0), 10)  # 画圆心

    cv2.imshow('', coins_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def test_hough_circle2():
    img = cv2.imread('./images/weiqi.png', 0)
    gray = cv2.medianBlur(img, 5)
    cv2.imshow('filter', gray)

    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, minDist=10,
        param1=100, param2=55).squeeze()

    circles = np.uint16(np.around(circles))
    img3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    print(circles)
    for i in circles:
        cv2.circle(img3, (i[0], i[1]), i[2], (0, 0, 255), 5)   # 画圆
        cv2.circle(img3, (i[0], i[1]), 2, (0, 255, 0), 10)     # 画圆心

    cv2.imshow('result', img3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()