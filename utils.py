import cv2

import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def crop_anhthe(img):
    """ cat vi tri so the"""
    img_d = img[300:420, 0:175]
    # cv2.imshow('Vi tri so the', img_d)
    # cv2.imwrite('anh_test.png',img_d)
    return img_d


def nhapchu(img, img2, tenanh):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_adaptive = img_gray
    img_adaptive = cv2.medianBlur(img_adaptive, 5)
    img_adaptive = cv2.GaussianBlur(img_adaptive, (5, 5), 0)
    img_adaptive = cv2.adaptiveThreshold(img_adaptive, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21,
                                         1)
    _, img_adaptive = cv2.threshold(img_adaptive, 150, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
    #

    cv2.imshow('anh_mssv', img_adaptive)
    # k = pytesseract.image_to_string(img_adaptive,config='--psm 6')
    k = pytesseract.image_to_string(img_adaptive, lang='eng',
                                    config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789')
    if len(k) <= 3:
        k1 = pytesseract.image_to_string(img2, lang='eng',
                                         config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789')
        if len(k1) <= 3:
            print('Khong khong doc duoc mssv {0}!!!'.format(tenanh))
        else:
            print('Ma so SV: {0} trong anh {1}'.format(k[:-1], tenanh))
    else:
        print('Ma so SV: {0} trong anh {1}'.format(k[:-1], tenanh))


def processing_vitrithe(img):
    img1 = img.copy()
    img_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img_adap = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10)
    img_adap = cv2.GaussianBlur(img_adap, (3, 3), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 5))
    edges = cv2.dilate(img_adap, kernel, 2)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # cnt = sorted(cnts, key=cv2.contourArea, reverse=True)
    x, y, w, h = cv2.boundingRect(cnts[0])
    # cv2.rectangle(img, (x, y), (x + w, y + h ), (255, 0, 0), 1)
    # cv2.imshow('anhdffdf',img)
    img_crop = img[y:y + h, x:x + w]
    return img_crop


def processing(img):
    img = cv2.pyrUp(img)
    img1 = img.copy()
    img_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('Anh_gray',img_gray)
    img_adap1 = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 17, 4)

    img_adap = img_adap1.copy()
    # cv2.imshow('anh_adap',img_adap)
    img_adap = cv2.GaussianBlur(img_adap, (7, 7), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 5))
    edges = cv2.dilate(img_adap, kernel, 1)
    # cv2.imshow('contour', edges)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = sorted(cnts, key=cv2.contourArea, reverse=True)
    x, y, w, h = cv2.boundingRect(cnt[0])

    # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 1)
    # cv2.imshow('ve contor',img)
    # img_crop
    img_crop = img[y:y + h - 100, x + 140:x + w]
    # cv2.imshow('crop_vitrithe',img_crop)
    return img_crop, img


def check1(img, tenanh):
    # img = cv2.imread('anhthe.png')
    img_crop = crop_anhthe(img)
    img_vitri_mssv, img_vitri2 = processing(img_crop)
    cv2.imshow('cropanh', img_crop)
    cv2.imshow('citri1', img_vitri_mssv)
    cv2.imshow('virt2', img_vitri2)
    img_mssv = processing_vitrithe(img_vitri_mssv)
    # print(img_mssv.shape)
    count = img_mssv.shape[0]
    if count >= 30:
        cv2.imshow('vi tri chua mssv', img_mssv)
        # m.imread_result(img_mssv)

        nhapchu(img_mssv, img_vitri2, tenanh)
    else:
        nhapchu(img_vitri_mssv, img_vitri2, tenanh)
    # cv2.waitKey(0)
# check1()
# img=cv2.imread('anhthe.png')
# check1(img)
