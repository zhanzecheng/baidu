import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
from collections import Counter
import glob

#you may change the directory
SAVE_FOLD = '/d_2t/backgroundtest/'
DATA_FOLD = ''

#a fast way to recognize the background color
def change_blackground(filename):
    img = cv2.imread(filename)
    m, n, _= img.shape
    for k in range(3):
        data = []
        data.append(img[0, 0, k])
        data.append(img[0, -1, k])
        data.append(img[-1, 0, k])
        data.append(img[-1, -1, k])
        data.append(img[0, n // 2, k])
        data.append(img[-1, n // 2, k])
        data.append(img[m // 2, -1, k])
        data.append(img[m // 2, 0, k])
        word_counts = Counter(data)
        number = word_counts.most_common(1)
        tmp = img[:, :, k]
        tmp[tmp == number[0][0]] = 0
        img[:, :, k] = tmp
    filename = filename.split('/')[-1]
    save_filename = SAVE_FOLD + filename
    cv2.imwrite(save_filename, img)

def cmp_key(x, y):
    x_tmp = int(x.split('/')[-1].split('.')[0])
    y_tmp = int(y.split('/')[-1].split('.')[0])
    if x_tmp < y_tmp:
        return -1
    elif x_tmp > y_tmp:
        return 1
    else:
        return 0

if __name__ == '__main__':
    filenames = glob.glob(DATA_FOLD)
    # filenames = sorted(filenames, cmp_key)
    print len(filenames)
    for file_tmp in filenames:
        change_blackground(file_tmp)
    print 'done'
