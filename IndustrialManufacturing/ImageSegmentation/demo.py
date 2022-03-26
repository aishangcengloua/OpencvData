import matplotlib.pyplot as plt
import cv2
import numpy as np
# 求两个点的差值
def getGrayDiff(image, currentPoint, tmpPoint):
    return abs(int(image[currentPoint[0], currentPoint[1]]) - int(image[tmpPoint[0], tmpPoint[1]]))

# 区域生长算法
def regional_growth(gray, seeds, threshold=5):
    # 每次区域生长的时候的像素之间的八个邻接点
    connects = [(-1, -1), (0, -1), (1, -1), (1, 0), \
                (1, 1), (0, 1), (-1, 1), (-1, 0)]

    threshold = threshold  # 生长时候的相似性阈值，默认即灰度级不相差超过15以内的都算为相同
    height, weight = gray.shape
    seedMark = np.zeros(gray.shape)
    seedList = []
    for seed in seeds:
        if (seed[0] < gray.shape[0] and seed[1] < gray.shape[1] and seed[0] > 0 and seed[1] > 0):
            seedList.append(seed)  # 将添加到的列表中
    print(seedList)
    label = 1  # 标记点的flag
    while (len(seedList) > 0):  # 如果列表里还存在点
        currentPoint = seedList.pop(0)  # 将最前面的那个抛出
        seedMark[currentPoint[0], currentPoint[1]] = label  # 将对应位置的点标志为1
        for i in range(8):  # 对这个点周围的8个点一次进行相似性判断
            tmpX = currentPoint[0] + connects[i][0]
            tmpY = currentPoint[1] + connects[i][1]
            if tmpX < 0 or tmpY < 0 or tmpX >= height or tmpY >= weight:  # 如果超出限定的阈值范围
                continue  # 跳过并继续
            grayDiff = getGrayDiff(gray, currentPoint, (tmpX, tmpY))  # 计算此点与像素点的灰度级之差
            if grayDiff < threshold and seedMark[tmpX, tmpY] == 0:
                seedMark[tmpX, tmpY] = label
                seedList.append((tmpX, tmpY))
    return seedMark

# 初始种子选择
def originalSeed(gray):
    ret, img1 = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY)  # 二值图，种子区域(不同划分可获得不同种子)
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(img1)  # 进行连通域操作，取其质点
    centroids = centroids.astype(int)
    return centroids

def test_region_grow():
    img = cv2.imread('./images/region_grow_2.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    seed = originalSeed(img)
    img = regional_growth(img, seed)

    # 用来正常显示中文标签
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.figure(figsize=(10, 5))
    plt.subplot(111),
    plt.imshow(img, cmap='gray'),
    plt.title('区域生长以后'),
    plt.axis("off")
    plt.show()

def test_kmeans_segmentation(infile):
    image = cv2.imread(infile)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(image.shape)

    pixel_values = image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    k = 4
    compactness, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    print(labels.shape)
    labels = labels.flatten()

    # convert all pixels to the color of the centroids
    segmented_image = centers[labels]
    print(segmented_image.shape)

    # reshape back to the original image dimension
    segmented_image = segmented_image.reshape(image.shape)

    # show the image
    plt.imshow(image)
    plt.title('input image')

    plt.figure()
    plt.imshow(segmented_image)
    plt.title('segmented image')

    # disable only the cluster number 2 (turn the pixel into black)
    masked_image = np.copy(image)
    # convert to the shape of a vector of pixel values
    masked_image = masked_image.reshape((-1, 3))
    # color (i.e cluster) to disable
    cluster = 2
    masked_image[labels == cluster] = [0, 0, 0]

    # convert back to original shape
    masked_image = masked_image.reshape(image.shape)
    # show the image
    plt.figure()
    plt.imshow(masked_image)
    plt.title('masked image')
    plt.show()

def test_watershed():
    img = cv2.imread('./images/coins.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cv2.imshow("thresh", thresh)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    cv2.imshow("opening", opening)

    sure_bg = cv2.dilate(opening, kernel, iterations=2)  # sure background area
    cv2.imshow("sure_bg", sure_bg)

    # Perform the distance transform algorithm
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    print(dist_transform)
    cv2.imshow('dist_transform1', dist_transform)
    # Normalize the distance image for range = {0.0, 1.0}
    cv2.normalize(dist_transform, dist_transform, 0, 1.0, cv2.NORM_MINMAX)

    # Finding sure foreground area
    ret, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
    print(dist_transform.max())
    cv2.imshow('dist_transform', dist_transform)
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)

    #不确定的区域是边界处
    unknown = cv2.subtract(sure_bg, sure_fg)
    cv2.imshow("sure_fg", sure_fg)
    cv2.imshow("unknown", unknown)

    # cv2.connectedComponents进行标记不同区域
    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    print(np.unique(markers))

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1
    cv2.imshow('11', markers.astype(np.uint8))

    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0
    plt.imshow(markers.astype(np.uint8), cmap = 'gray')
    plt.show()

    cv2.imshow("markers", markers.astype(np.uint8))

    markers_copy = markers.copy()
    markers_copy[markers == 0] = 150  # 灰色表示背景
    markers_copy[markers == 1] = 0  # 黑色表示背景
    markers_copy[markers > 1] = 255  # 白色表示前景
    markers_copy = np.uint8(markers_copy)

    # 使用分水岭算法执行基于标记的图像分割，将图像中的对象与背景分离
    markers = cv2.watershed(img, markers)

    img[markers == -1] = [0, 0, 255]  # 将边界标记为红色

    cv2.imshow("result", img)
    cv2.waitKey(-1)

def test_grab_cut():
    src = cv2.imread("./images/flower.png")
    src = cv2.resize(src, (0,0), fx=0.5, fy=0.5)
    r = cv2.selectROI('input', src, False, False)  # 返回 (x_min, y_min, w, h)

    # roi区域
    roi = src[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]

    # 原图mask
    mask = np.zeros(src.shape[:2], dtype=np.uint8)

    # 矩形roi
    rect = (int(r[0]), int(r[1]), int(r[2]), int(r[3])) # 包括前景的矩形，格式为(x,y,w,h)

    bgdmodel = np.zeros((1,65),np.float64) # bg模型的临时数组
    fgdmodel = np.zeros((1,65),np.float64) # fg模型的临时数组

    cv2.grabCut(src,mask,rect,bgdmodel,fgdmodel, 11, mode=cv2.GC_INIT_WITH_RECT)

    # 提取前景和可能的前景区域
    mask2 = np.where((mask==1) + (mask==3), 255, 0).astype('uint8')

    print(mask2.shape)

    result = cv2.bitwise_and(src,src,mask=mask2)
    cv2.imshow('roi', roi)
    cv2.imshow("result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    test_kmeans_segmentation('./images/shenzhen.png')
    # test_grab_cut()
    # test_watershed()