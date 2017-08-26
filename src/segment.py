# coding:utf8

import sys
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

# you may change the directory
SAVE_FLOD = '/d_2t/segmenttest/'
DATA_FLOD = ''

COMPLEMENT = 1


def preprocess(gray):
    # 1. Sobel算子，x方向求梯度
    # element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # gray = cv2.erode(gray, element1)
    # gray = cv2.dilate(gray, element1)
    gray = cv2.medianBlur(gray, 9)
    sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize = 3)
    # 2. 二值化
    ret, binary = cv2.threshold(sobel, 0, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY)
    # binary = cv2.medianBlur(binary, 3)
    # 3. 膨胀和腐蚀操作的核函数
    element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (26, 9))
    element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (24, 7))

    # 4. 膨胀一次，让轮廓突出
    dilation = cv2.dilate(binary, element2, iterations = 1)

    # 5. 腐蚀一次，去掉细节，如表格线等。注意这里去掉的是竖直的线
    # erosion = cv2.erode(dilation, element1, iterations = 1)

    # 6. 再次膨胀，让轮廓明显一些
    dilation2 = cv2.dilate(dilation, element2, iterations = 3)

    # 7. 存储中间图片

    return dilation2


def findTextRegion(img):
    region = []

    # 1. 查找轮廓
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 2. 筛选那些面积小的
    for i in range(len(contours)):
        cnt = contours[i]
        # 计算该轮廓的面积
        area = cv2.contourArea(cnt)

        # 面积小的都筛选掉
        if(area < 5000):
            continue

        # 轮廓近似，作用很小
        epsilon = 0.001 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # 找到最小的矩形，该矩形可能有方向
        rect = cv2.minAreaRect(cnt)


        # box是四个点的坐标
        box = cv2.cv.BoxPoints(rect)
        box = np.int0(box)
        print "box is: "
        print box
        # print 'box is', box
        # 计算高和宽
        box[0][0] -= COMPLEMENT
        if box[0][0] < 0:
            box[0][0] = 0
        box[1][0] -= COMPLEMENT
        if box[1][0] < 0:
            box[1][0] = 0
        box[0][1] += COMPLEMENT
        box[1][1] -= COMPLEMENT
        box[2][0] += COMPLEMENT
        box[3][0] += COMPLEMENT
        box[2][1] -= COMPLEMENT
        box[3][1] += COMPLEMENT
        height = abs(box[0][1] - box[2][1])
        width = abs(box[0][0] - box[2][0])

        # 筛选那些太细的矩形，留下扁的
        if(height > width * 1.2):
            continue


        region.append(box)

    return region

def cmp_key(box1, box2):
    if box1[0][1] < box2[0][1]:
        return -1
    elif box1[0][1] > box2[0][1] :
        return 1
    else:
        return 0


def detect(imgPath):
    # 1.  转化成灰度图
    img = cv2.imread(imgPath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. 形态学变换的预处理，得到可以查找矩形的图片
    dilation = preprocess(gray)

    # 3. 查找和筛选文字区域
    region = findTextRegion(dilation)
    # assert len(region) <= 3
    # 4. 用绿线画出这些找到的轮廓
    filename = imgPath.split('/')[-1].split('.')[0]
    # count = 0
    # for i, box in enumerate(region):
    #     height = [box[0][0], box[1][0], box[2][0], box[3][0]]
    #     width = [box[0][1], box[1][1], box[2][1], box[3][1]]
    #     height = [box[0][0], box[1][0], box[2][0], box[3][0]]
    #     width = [box[0][1], box[1][1], box[2][1], box[3][1]]
    #     tmp_img = img[min(width): max(width), min(height): max(height),0]
    #     hist = cv2.calcHist([tmp_img],
    #                         [0],  # 使用的通道
    #                         None,  # 没有使用mask
    #                         [256],  # HistSize
    #                         [0.0, 255.0])  # 直方图柱的范围
    #
    #     hist[hist.argmax()] = 0
    #     tmp_max = 0
    #     for ii in range(16):
    #         tmp = 0
    #         for j in range(ii * 16 + 1, (ii + 1) * 16 - 1):
    #             if hist[j - 1] >= 4 and hist[j + 1] >= 4:
    #                 tmp += hist[j]
    #         if tmp > tmp_max:
    #             tmp_max = tmp
    #     print 'the tmp_max is', tmp_max
    #     if tmp_max < 300:
    #        continue
    #     cv2.imwrite('C:/Users/Administrator/Desktop/result/{0}_{1}.png'.format(filename, count),
    #                 img[min(width): max(width), min(height): max(height)])
    #     count += 1
    # assert count <= 3
    region = sorted(region, cmp_key)
    if len(region) > 2:
        count = 0
        for i, box in enumerate(region):
            height = [box[0][0], box[1][0], box[2][0], box[3][0]]
            width = [box[0][1], box[1][1], box[2][1], box[3][1]]
            height = [box[0][0], box[1][0], box[2][0], box[3][0]]
            width = [box[0][1], box[1][1], box[2][1], box[3][1]]
            tmp_img = img[min(width): max(width), min(height): max(height), 0]
            hist = cv2.calcHist([tmp_img],
                                [0],  # 使用的通道
                                None,  # 没有使用mask
                                [256],  # HistSize
                                [0.0, 255.0])  # 直方图柱的范围

            hist[hist.argmax()] = 0
            tmp_max = 0
            for ii in range(16):
                tmp = 0
                for j in range(ii * 16 + 1, (ii + 1) * 16 - 1):
                    if hist[j - 1] >= 4 and hist[j + 1] >= 4:
                        tmp += hist[j]
                if tmp > tmp_max:
                    tmp_max = tmp
            print   'the tmp_max is', tmp_max
            if tmp_max < 150:
                continue
            cv2.imwrite(SAVE_FLOD + '{0}_{1}.png'.format(filename, count),
                        img[min(width): max(width), min(height): max(height)])
            count += 1
        assert count <= 3
    else:
        for i, box in enumerate(region):
            # cv2.drawContours(img, [box], 0, (0, 255, 0), 2)
            height = [box[0][0], box[1][0], box[2][0], box[3][0]]
            width = [box[0][1], box[1][1], box[2][1], box[3][1]]
            # print img.shape
            # print min(height), max(height)
            cv2.imwrite(SAVE_FLOD + '{0}_{1}.png'.format(filename, i),
                        img[min(width): max(width), min(height): max(height)])
        # cv2.namedWindow("img", cv2.WINDOW_NORMAL)
        # cv2.imshow("img", img)
        #
        # # 带轮廓的图片
        # cv2.imwrite("contours.png", img)
        # # hist = cv2.calcHist([img],
        # #                     [0],  # 使用的通道
        # #                     None,  # 没有使用mask
        # #                     [256],  # HistSize
        # #                     [0.0, 255.0])  # 直方图柱的范围
        #
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


if __name__ == '__main__':

    filenames = glob.glob(DATA_FLOD)
    for filename in filenames:
        print filename
        detect(filename)


