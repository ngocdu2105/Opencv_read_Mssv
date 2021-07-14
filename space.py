import cv2
import numpy as np
import utils as u


def getTranform(approx,img):
    '''approx of contours'''
    w, h = 640, 480
    ps1 = np.float32([approx[0][0],approx[2][0],approx[1][0],approx[3][0]]).reshape(-1, 1, 2)
    ps2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]]).reshape(-1, 1, 2)
    M = cv2.getPerspectiveTransform(ps1, ps2)
    dst = cv2.warpPerspective(img, M, (w, h))
    cv2.imwrite('anhthe.png',dst)
    cv2.imshow('anh',dst)
    return dst
def smooth_blur(img):

    #
    #
    kernel =cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    img_adaptive = cv2.bilateralFilter(img, 11, 65, 65)

    # k= cv2.erode(img_adaptive, kernel, iterations=1)
    k=cv2.dilate(img_adaptive,kernel,iterations=1)
    cv2.imshow('dilate', k)
    # k=cv2.erode(k,kernel,iterations=1)
    # cv2.imshow('dilate', k)
    k=cv2.GaussianBlur(k,(7,7),2)
    # k=cv2.medianBlur(k,7)
    cv2.imshow('gaussian',k) #150,200
    # find_edges(k, kernel, img)
    return k
def find_edges(k):
    #150
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    img_canny=cv2.Canny(k,50,150)
    # img_canny=
    img_canny = cv2.dilate(img_canny, kernel, iterations=1)
    # img_grauss =cv2.medianBlur(img_canny,5)
    # findcontour(img_canny,img)ff
    # hough_line(img_canny,img)
    cv2.imshow('df',img_canny)
    cv2.imshow('erode',k)
    # cv2.imshow('dff',)
    return img_canny

def findcontour(edge,img):
    contours, _ =cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area =0
    for i in contours:
        for i in contours:
            area = cv2.contourArea(i)
            if area > 700:
                peri = cv2.arcLength(i, True)
                approx = cv2.approxPolyDP(i, 0.02 * peri, True)
                if area > max_area and len(approx) == 4:
                    biggest = approx
                    max_area = area
        # np.float32([[775, 707], [1368, 831], [801, 1200], [1444, 1250]])
    cv2.drawContours(img, [approx], -1, (255, 0, 0), 2)
    # print((approx))
    # approx=sort_approx(approx)
    # getTranform(approx, img)
    cv2.imshow('contours', img)
    return approx
def sort_approx(approx):
    # print(2*approx)
    k=sorted(approx,key=lambda x:(x[0][0],x[0][1]))
    return k
def test():
    img_1 = cv2.imread('Anh/anhthe3.jpg')

    img_1 = cv2.resize(img_1, None, fx=0.5, fy=0.5)
    img = img_1
    img_blur=smooth_blur(img)
    # img_blur=main.morphologyEx(img_blur,1)
    # img_blur=
    edges= find_edges(img_blur)
    approx=findcontour(edges,img)
    approx = sort_approx(approx)
    dst=getTranform(approx, img)
    u.check1(dst)
    cv2.waitKey(0)




test()
