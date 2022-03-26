import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def test_watershed() :
    image = cv.imread('images/coins.jpg')
    image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    #基于直方图的二值化处理
    _, thresh = cv.threshold(image_gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

    #做开操作，是为了除去白噪声
    kernel = np.ones((3, 3), dtype = np.uint8)
    opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations = 2)

    #做膨胀操作，是为了让前景漫延到背景，让确定的背景出现
    sure_bg = cv.dilate(opening, kernel, iterations = 2)

    #为了求得确定的前景，也就是注水处使用距离的方法转化
    dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
    #归一化所求的距离转换，转化范围是[0, 1]
    cv.normalize(dist_transform, dist_transform, 0, 1.0, cv.NORM_MINMAX)
    #再次做二值化，得到确定的前景
    _, sure_fg = cv.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    #得到不确定区域也就是边界所在区域，用确定的背景图减去确定的前景图
    unknow = cv.subtract(sure_bg, sure_fg)

    #给确定的注水位置进行标上标签，背景图标为0，其他的区域由1开始按顺序进行标
    _, markers = cv.connectedComponents(sure_fg)

    # cv.imshow('markers', markers.astype(np.uint8))
    # cv.waitKey(0)

    #让标签加1，这是因为在分水岭算法中，会将标签为0的区域当作边界区域（不确定区域）
    markers += 1

    #是上面所求的不确定区域标上0
    markers[unknow == 255] = 0
    # print(markers.dtype)  int32
    markers_copy = markers.copy()

    # 使用分水岭算法执行基于标记的图像分割，将图像中的对象与背景分离
    markers = cv.watershed(image, markers)

    #分水岭算法得到的边界点的像素值为-1
    image[markers == -1] = [0, 0, 255]

    images = [thresh, opening, sure_bg, dist_transform, sure_fg, unknow, markers_copy, image]

    titles = ['thresh', 'opening', 'sure_bg', 'dist_tranform', 'sure_fg', 'unknow', 'markers', 'image']
    plt.figure(figsize = (8, 6.1))

    for i in range(len(images)) :
        if i == 7 :
            plt.subplot(2, 4, i + 1)
            plt.imshow(cv.cvtColor(images[i], cv.COLOR_BGR2RGB))
            plt.title(titles[i])
            plt.axis('off')
        else :
            plt.subplot(2, 4, i + 1)
            plt.imshow(images[i], cmap = 'gray')
            plt.title(titles[i])
            plt.axis('off')
    plt.tight_layout()
    plt.savefig('figure.png')
    plt.show()

if __name__ == '__main__':
    test_watershed()