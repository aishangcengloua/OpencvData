import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import cv2
 
# Good for the b/w test images used
MIN_CANNY_THRESHOLD = 10
MAX_CANNY_THRESHOLD = 50
    
def gradient_orientation(image):
    '''
    Calculate the gradient orientation for edge point in the image
    '''
    dx = cv2.Sobel(image, -1, dx=1, dy=0 )
    dy = cv2.Sobel(image, -1, dx=0, dy=1)
    gradient = np.arctan2(dy,dx) * 180 / np.pi
    
    return gradient
    
def build_r_table(image, origin):
    '''
    Build the R-table from the given shape image and a reference point
    '''
    edges = cv2.Canny(image, MIN_CANNY_THRESHOLD, MAX_CANNY_THRESHOLD)
    gradient = gradient_orientation(edges)
    
    r_table = defaultdict(list)
    for (i,j),value in np.ndenumerate(edges):
        if value:
            angle = gradient[i,j]
            r_table[angle].append((origin[0]-i, origin[1]-j))
 
    return r_table
 
def accumulate_gradients(r_table, grayImage):
    '''
    Perform a General Hough Transform with the given image and R-table
    '''
    edges = cv2.Canny(grayImage, MIN_CANNY_THRESHOLD, MAX_CANNY_THRESHOLD)
    gradient = gradient_orientation(edges)
    
    accumulator = np.zeros(grayImage.shape)
    for (i,j),value in np.ndenumerate(edges):
        if value:
            table_entries = r_table[gradient[i,j]]
            for r in table_entries:
                accum_i, accum_j = i+r[0], j+r[1]
                if accum_i < accumulator.shape[0] and accum_j < accumulator.shape[1]:
                    accumulator[int(accum_i), int(accum_j)] += 1
                    
    return accumulator
 
def general_hough_closure(reference_image):

    referencePoint = (reference_image.shape[0]/2, reference_image.shape[1]/2)
    r_table = build_r_table(reference_image, referencePoint)
    
    def f(query_image):
        return accumulate_gradients(r_table, query_image)
        
    return f
 
def n_max(a, n):
    '''
    Return the N max elements and indices in a
    '''
    indices = a.ravel().argsort()[-n:]
    indices = (np.unravel_index(i, a.shape) for i in indices)
    return [(a[i], i) for i in indices]
 
def test_general_hough(gh, reference_image, query):
    '''
    Uses a GH closure to detect shapes in an image and create nice output
    '''
    query_image = cv2.imread(query, cv2.IMREAD_GRAYSCALE)
    accumulator = gh(query_image)
     
    fig = plt.figure()
    plt.gray()
    fig.add_subplot(2,2,1)
    plt.title('Reference image')
    plt.imshow(reference_image)
    
    fig.add_subplot(2,2,2)
    plt.title('Query image')
    plt.imshow(query_image)
    
    fig.add_subplot(2,2,3)
    plt.title('Accumulator')
    plt.imshow(accumulator)
    
    fig.add_subplot(2,2,4)
    plt.title('Detection')
    plt.imshow(query_image)
    
    # top 5 results in red
    m = n_max(accumulator, 5)
    y_points = [pt[1][0] for pt in m]
    x_points = [pt[1][1] for pt in m] 
    plt.scatter(x_points, y_points, marker='o', color='r')
 
    # top result in yellow
    i,j = np.unravel_index(accumulator.argmax(), accumulator.shape)
    plt.scatter([j], [i], marker='x', color='y')
     
    plt.show()
    
    return
 
def test():
    reference_image = cv2.imread("./images/ght_images/s.png", cv2.IMREAD_GRAYSCALE)
    detector = general_hough_closure(reference_image)
    test_general_hough(detector, reference_image, "./images/ght_images/s_test.png")

    reference_image = cv2.imread("./images/ght_images/diamond.png", cv2.IMREAD_GRAYSCALE)
    detect_s = general_hough_closure(reference_image)
    # test_general_hough(detect_s, reference_image, "./images/ght_images/diamond_test1.png")
    # test_general_hough(detect_s, reference_image, "./images/ght_images/diamond_test2.png")
    # test_general_hough(detect_s, reference_image, "./images/ght_images/diamond_test3.png")
    # test_general_hough(detect_s, reference_image, "./images/ght_images/diamond_test4.png")
    # test_general_hough(detect_s, reference_image, "./images/ght_images/diamond_test5.png")
    test_general_hough(detect_s, reference_image, "./images/ght_images/diamond_test6.png")
