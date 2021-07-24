import glob
import alth as a
import cv2
import numpy as np
import os
import utils as u


def find_object_flann(img_vao):
    MIN_MATCH_COUNT = 10
    img1 = cv2.imread('sample/sanple_anhthe.jpg')  # Index image
    img1 = cv2.resize(img1, dsize=(640, 480))  # img2 = cv2.pyrDown(img_vao) # training image
    img2 = img_vao
    # Initialize the SIFT detector
    sift = cv2.SIFT_create()
    # Use SIFT to find key points and descriptors

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    ## Store all matching items that meet the conditions according to Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        # matchesMask = mask.ravel().tolist()
        h, w = img1.shape[:2]
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)

        # print(dst)
        # dst=a.sort_approx(dst)
        # anhchuyen=a.getTranform(dst,img2)
        # u.check1(anhchuyen)
        # print('dst:',dst)
        # img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
        cv2.imshow('Anh', img2)
        return dst
    else:
        print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
        matchesMask = None

    # draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
    #                    singlePointColor=None,
    #                    matchesMask=matchesMask,  # draw only inliers
    #                    flags=2)
    # img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
    # cv2.imshow('gray', img3)


def load_images_from_folder(folder):
    ext = ['png', 'jpg']
    files = []
    try:
        [files.extend(glob.glob(folder + str('/') + '*.' + e)) for e in ext]
        # images = [cv2.imread(files) for files in files]
        return files

    except:
        print('Loi doc anh vao:')
        return None


def imread_result(img, k=0, path='result'):
    while True:
        if os.path.isfile(path + str('/') + f'anh_result_{k:04}.png') is False:
            cv2.imwrite(path + str('/') + f'anh_result_{k:04}.png', img)
            break
        else:
            k += 1


def read_nameImg(Path):
    names_file = []
    for path in Path:
        filename = os.path.basename(path)
        names_file.append(filename)
    return names_file


ten_fileImg = ['anhthe', 'Anh']
for ten in ten_fileImg:
    names_file = load_images_from_folder(ten)
    name_file = read_nameImg(names_file)
    for index, i in enumerate(names_file):
        img = cv2.imread(i)
        if img is None:
            print('Khong doc duoc anh: ', name_file[index])
            continue
        else:
            img = cv2.pyrDown(img)

            dst = find_object_flann(img)
            dst = a.sort_approx(dst)
            anhchuyen = a.getTranform(dst, img, name_file[index])
            # imread_result(anhchuyen)
            if anhchuyen is None:
                pass
            else:
                try:
                    u.check1(anhchuyen, name_file[index])
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                except:
                    print('Loi anh {0} khi cat vi tri mssv '.format(name_file[index]))

''' tỉ lệ đọc ảnh 100%, nhận diện chữ 73% ,
 tỉ lệ nhận diện đúng chữ  81,4%.  
 chữ không nhận diện được đa phần ảnh kém chất lượng không cải thiện được trong tổng 37 ảnh'''
