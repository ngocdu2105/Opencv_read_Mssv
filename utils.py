import os
import alth as a
import cv2

import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def crop_anhthe(img):
    ''' cat vi tri so the [0:420,0:180]'''
    img_d = img[330:410, 80:180]
    # cv2.imshow('Vi tri so the', img_d)
    # cv2.imwrite('anh_test.png',img_d)
    return img_d

def nhapchu(img, tenanh):
    img = cv2.resize(img, dsize=(155, 42))
    mssv=tenanh.split('_')[0]
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_adaptive = img_gray
    # img_adaptive = cv2.medianBlur(img_adaptive, 5)
    img_adaptive = cv2.GaussianBlur(img_adaptive, (5,5), 0)
    img_adaptive = cv2.adaptiveThreshold(img_adaptive, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19,2)
    #21,2
    _, img_adaptive = cv2.threshold(img_adaptive, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)

    # cv2.imshow('anh_mssv', img_adaptive)
    k = pytesseract.image_to_string(img_adaptive, lang='eng',
                                    config='--psm 10 --oem 3 -c tessedit_char_whitelist=1802345679')
    if len(k) <= 3:
            print('Khong khong doc duoc mssv {0}!!!'.format(tenanh))
            cv2.imwrite('result/'+str('L_')+mssv+str('.png'),img)
    else:
        k=k.strip()
        print('Ma so SV: {0} trong anh {1}'.format(k, tenanh))
        if str(mssv) !=str(k):
            cv2.imwrite('result/' +k +str('_') + mssv+str('.png'), img)


def processing(img):
    # img = cv2.pyrUp(img)

    img1 = img.copy()
    img1=a.img_bilateralFilter(img1)
    img_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('Anh_gray', img_gray)
    img_adap1 = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 2)
    img_adap1=cv2.medianBlur(img_adap1,5)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21,3))
    img_adap1 = cv2.dilate(img_adap1, kernel, 2)
    img_adap1=a.erode(img_adap1,5)
    img_adap1=a.morphologyEx(img_adap1,0)
    cnts, _ = cv2.findContours(img_adap1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = sorted(cnts, key=cv2.contourArea, reverse=True)
    x, y, w, h = cv2.boundingRect(cnt[1])
    img_crop=img[y-6:y+h+3,x:x+w+3]


    return img_crop
def imread_result(img, k=0, path='result'):
    while True:
        if os.path.isfile(path + str('/') + f'anh_result_{k:04}.png') is False:
            cv2.imwrite(path + str('/') + f'anh_result_{k:04}.png', img)
            break
        else:
            k += 1

def check1(img, tenanh):
    img_crop = crop_anhthe(img)
    img_vitri_mssv = processing(img_crop)


    nhapchu(img_vitri_mssv, tenanh)
    # cv2.waitKey(0)
