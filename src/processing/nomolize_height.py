# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 13:19:35 2017

@author: Administrator
"""
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt

# you may change the directory
SMALL_DATA_FLOD = ''
BIG_DATA_FLOD = ''

# for filename in FILENAMES:
#     img = cv2.imread(filename)
#     img = img[:,:240,:]
#     cv2.imwrite(filename, img)


def get_the_size(filename):
    img2 = cv2.imread(filename)
    try:
        img = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    except:
        print filename
        quit()
    img = cv2.medianBlur(img, 9)
    width, height = img.shape
    count_end = height
    for i in range(0, height)[::-1]:
        if len(img[:, i][img[:, i] >= 100]) < 8:
            count_end -= 1
        else:
            break
    count_begin = 0
    for i in range(0, height):
        if len(img[:, i][img[:, i] >= 100]) < 8:
            count_begin += 1
        else:
            break
    assert count_begin <= count_end
    if count_begin - 5 > 0 and count_end + 5 < height:
        img = img2[:, count_begin - 5:count_end + 5, :]
    elif count_begin - 5 <= 0 and count_end + 5 < height:
        img = img2[:, :count_end + 5, :]
    elif count_begin - 5 > 0 and count_end + 5 > height:
        img = img2[:, count_begin - 5:, :]
    else:
        pass
    # filename = filename.replace('big', 'big')
    cv2.imwrite(filename, img)
if __name__ == '__main__':

    FILENAMES = glob.glob(SMALL_DATA_FLOD)
    print len(FILENAMES)
    for filename in FILENAMES:
        get_the_size(filename)
    print 'done the samll'

    FILENAMES = glob.glob(BIG_DATA_FLOD)
    print len(FILENAMES)
    for filename in FILENAMES:
        get_the_size(filename)
    print 'done the samll'
