import cv2
import numpy as np
def morphologyEx(img, close=1):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    if close == 1:
        img_ex = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    else:
        img_ex = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    # img_ex =cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)
    return img_ex


def dilate(edges, iterations=1):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.dilate(edges, kernel, iterations)
    return edges


def erode(edges, iterations=1):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.erode(edges, kernel, iterations)
    return edges


def img_bilateralFilter(img, a=11, b=75):
    return cv2.bilateralFilter(img, a, b, b)


def img_medianBlur(img, ksize=5):
    return cv2.medianBlur(img, 5)


def img_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def img_canny(img, min, max):
    return cv2.Canny(img, min, min)


def hough_line(img, dst, rho=0.7, theta=np.pi / 180, thresh=400):
    # img=cv2.C
    lines = cv2.HoughLines(img, rho, theta, thresh)
    print(lines)
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(dst, (x1, y1), (x2, y2), (0, 0, 255), 1)
    # for line in lines:
    #     x1,y1,x2,y2 =line[0]
    #     cv2.line(img,(x1,y1),(x2,y2),(255,0,0),2)
    cv2.imshow('ifd', dst)


def adaptive(img, max=255, a=21, b=5):
    return cv2.adaptiveThreshold(img, max, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, a, b)
#|

def initializeTrackbars(intialTrackbarVals=0):
    cv2.namedWindow("Trackbar")
    cv2.resizeWindow("Trackbars", 360, 240)
    cv2.createTrackbar("Threshold1", "Trackbars", 200, 255, nothing)
    cv2.createTrackbar("Threshold2", "Trackbars", 200, 255, nothing)


def sort_approx(approx):
    # print(2*approx)
    k = sorted(approx, key=lambda x: (x[0][0]+ x[0][1]))
    if k[2][0][1] <k[1][0][1]:
        k=[k[0],k[2],k[1],k[3]]
    return k


def getTranform(approx, img,tenanh):
    '''approx cua '''
    w, h = 640, 480
    ps1 = np.float32(approx).reshape(-1, 1, 2)
    ps2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]]).reshape(-1, 1, 2)
    M = cv2.getPerspectiveTransform(ps1, ps2)
    dst = cv2.warpPerspective(img, M, (w, h))
    cv2.imshow('anh chuyen '+str(tenanh), dst)
    # cv2.imwrite('sample/sample.png',dst)
    return dst

def nothing(x):
    pass


def biggestContour(contours, s):
    biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > s:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    return biggest, max_area


def valTrackbars():
    Threshold1 = cv2.getTrackbarPos("Threshold1", "Trackbars")
    Threshold2 = cv2.getTrackbarPos("Threshold2", "Trackbars")
    src = Threshold1, Threshold2
    return src


def canny_gray(img, min, max):
    '''Dedine edges'''
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(img_gray, min, max)
    return edged


def drawing_contour(edged, img, defautl=-1):
    ''' Drawing contours '''
    contours, hierarchy = cv2.findContours(edged,
                                           cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if defautl == -1:
        cv2.drawContours(img, contours, defautl, (0, 255, 0), 3)
    else:
        return contours


def imshow(names, img, d=1):
    cv2.imshow(names, img)
    if d == 2:
        waikey(0)


def img_black(height, width):
    img = np.zeros((height, width, 3), np.uint8)
    return img


def img_erode(img, k=6):
    kernel = kernel = np.ones((k, k), np.uint8)
    img = cv2.erode(img, kernel, cv2.BORDER_REFLECT)
    return img


def waikey(default=1):
    if default == 1:
        cv2.waitKey(0)


def imread(s, default=1):
    img = cv2.imread(s)
    if default == 1:
        return img
    else:
        img1 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return img1


def mask_img_remove(clustedImage, img1, lables, cluster=0):
    masked_image = np.copy(clustedImage)
    masked_image = img1.reshape((-1, 3))
    # masked_image=np.float32(clustedImage)
    masked_image[lables.flatten() == cluster] = [255, 0, 0]
    masked_image = masked_image.reshape(img1.shape)
    cv2.imshow('remove cluters', masked_image)


def kmean(reshapedImaged, original, numberOfClusters=2):
    stopCriteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
    ret, lables, clusters = cv2.kmeans(reshapedImaged, numberOfClusters, None, stopCriteria, 10,
                                       cv2.KMEANS_RANDOM_CENTERS)
    clusters = np.uint8(clusters)
    print("clusters", clusters)
    intermediateImage = clusters[lables.flatten()]
    print("intermediteImage", lables.flatten())
    clustedImage = intermediateImage.reshape((original.shape))
    return clustedImage, lables
