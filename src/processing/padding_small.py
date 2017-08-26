# coding:utf-8
import glob
import cv2
import numpy as np


filenames = glob.glob('/d_2t/bigtesteq/*.png')
SAVE_FOLD = '/d_2t/bigtesteqPADDING_HEIGHT/'

PADDING_WIDTH = 80
PADDING_HEIGHT = 766
import cPickle as pickle

for filename in filenames:
    img =cv2.imread(filename)
    if True:
        width, height, depth = img.shape
        assert width <= PADDING_WIDTH
        if width < PADDING_WIDTH:
            dim = (PADDING_WIDTH - width) // 2
            img = np.pad(img, ((dim, dim), (0, 0), (0, 0)), 'constant', constant_values=(0, 0))
            if (PADDING_WIDTH - width) % 2 != 0:
                img = np.pad(img, ((0, 1), (0, 0), (0, 0)), 'constant', constant_values=(0, 0))
        # elif width > PADDING_WIDTH:
        #     img = img[1 : 65, :, :]

        if height < PADDING_HEIGHT:
            img = np.pad(img, ((0, 0), (0, PADDING_HEIGHT - height), (0, 0)), 'constant', constant_values=(0, 0))

        width, height, depth = img.shape
        assert width == PADDING_WIDTH
        print filename
        assert height == PADDING_HEIGHT

        filename = filename.split('/')[-1]
        cv2.imwrite(SAVE_FOLD + filename, img)


