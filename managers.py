import glob
import alth as a
import cv2
import numpy as np
import os
import utils as u
import time


def find_object_flann(img_vao):
    MIN_MATCH_COUNT = 2
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
        return dst
    else:
        print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
        matchesMask = None


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


def imread_result(img, k=0, path='models'):
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

def create_Folder(filepath='result'):
    import os
    if os.path.exists(filepath):
        file=os.listdir(filepath)
        for file in file:
            os.remove(filepath+str('/')+file)
    else:
        os.mkdir(filepath)
start_time = time.time()
#
create_Folder(filepath='models')
create_Folder()
ten_fileImg = ['anhthe', 'Anh']
countFile=0
loivt=0
for ten in ten_fileImg:
    names_file = load_images_from_folder(ten)
    countFile +=len(names_file)
    name_file = read_nameImg(names_file)
    for index, i in enumerate(names_file):
        img = cv2.imread(i)

        if img is None:
            print('Khong doc duoc anh: ', name_file[index])
            loivt +=1
            continue
        else:
            k=0
            h,w=img.shape[:2]
            if h+w >1800:
                if h < w:
                    k = int(567 * w / h)
                    img = cv2.resize(img, (k, 567))

                else:
                    k = int(567 * h / w)
                    img = cv2.resize(img, (567, k))

            # 756,567
            dst = find_object_flann(img)
            dst = a.sort_approx(dst)
            anhchuyen = a.getTranform(dst, img, name_file[index])

            if anhchuyen is None:
                pass
            else:
                try:
                    u.check1(anhchuyen, name_file[index])
                    # cv2.waitKey(0)
                    cv2.destroyAllWindows()
                except:
                    print('Loi anh {0} khi cat vi tri mssv '.format(name_file[index]))

                    imread_result(anhchuyen)
                    loivt +=1
ketqua=load_images_from_folder('result')
ten_ketqua=read_nameImg(ketqua)
maloi=0
for i in ten_ketqua:
    maloiCount,_=i.split('_')
    if maloiCount=='L':
        maloi+=1

accurat =float((len(ketqua)+loivt)/countFile)
ptkn=float(maloi/countFile)
ptocr=float((len(ketqua))/countFile)-ptkn
print('tỉ lệ sai:{0}'.format(accurat))
print('Ảnh không nhận dạng được {0} ảnh chiếm {1} tổng số ({2}/{3})'.format(maloi,ptkn,maloi,countFile))
print('Ảnh nhận dạng OCR sai {0} chiếm {1} tổng số ({2}/{3})'.format(len(ketqua)-maloi,ptocr,len(ketqua)-maloi,countFile))
print('Time: ',time.time()-start_time)
''' tỉ lệ đọc ảnh 100%, nhận diện chữ 73% ,
 tỉ lệ nhận diện đúng chữ  81,4%.  
 chữ không nhận diện được đa phần ảnh kém chất lượng không cải thiện được trong tổng 37 ảnh'''
