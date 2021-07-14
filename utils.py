import cv2
import numpy as np
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def crop_anhthe(img):
    ''' cat vi tri so the'''
    img_d = img[300:420, 0:175]
    cv2.imshow('anh', img_d)
    return img_d


def nhapchu(img):
    # if img==None:
    #     img = cv2.imread('anhthe.png')
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    k = pytesseract.image_to_string(img_gray)
    print(k)


def check1(img):
    # img = cv2.imread('anhthe.png')
    img_crop = crop_anhthe(img)
    nhapchu(img_crop)
    img_gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
    img_adap = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10)
    cv2.imshow('so the', img_gray)

    cv2.waitKey(0)
