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

count_error = 0
def get_the_size(filename):
    img2 = cv2.imread(filename)
    aa, bb, _ = img2.shape
    if bb > 9:
        img = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        img = cv2.medianBlur(img, 7)
        width, height = img.shape
        count_end = width
        for i in range(1, width - 1)[::-1]:
            if len(img[i, :][img[i, :] >= 100]) < 3:
                count_end -= 1
            else:
                break
        count_begin = 0
        for i in range(1, width):
            if len(img[i, :][img[i, :] >= 100]) < 3:
                count_begin += 1
            else:
                break
        print   count_begin, count_end, filename
        assert  count_begin < count_end
        if count_begin - 4 > 0 and count_end + 4 < width:
            img = img2[count_begin - 4:count_end + 4, :, :]
        elif count_begin - 4 <= 0 and count_end + 4 < width:
            img = img2[:count_end + 4, :, :]
        elif count_begin - 4 > 0 and count_end + 4 > width:
            img = img2[count_begin - 4:, :, :]
        else:
            img = img2
        #
    else:
        img = np.zeros((80,10,3))
    # filename = filename.replace('bigtest', 'bigtesteq')
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

    print count_error