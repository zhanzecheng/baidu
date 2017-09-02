# coding:utf-8
'''
我们这里使用双向lstm和CTC_loss构建模型
'''
from captcha.image import ImageCaptcha
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import glob
from sklearn.utils import shuffle
from src.processing.gen_captcha_frac import number
from src.processing.gen_captcha_frac import alphabet
from src.processing.gen_captcha_frac import ALPHABET
from src.processing.gen_captcha_frac import chinese_character
from keras import backend as K
K.set_image_dim_ordering('tf')
import random
import string
from keras.callbacks import ModelCheckpoint
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


FILE_NAMES = glob.glob('/d_2t/big_eq/*.png')
FILE_NAMES = shuffle(FILE_NAMES)
print len(FILE_NAMES)


characters = number + alphabet + ['_'] + chinese_character + ALPHABET
CHAR_SET_LEN = len(characters)
print CHAR_SET_LEN
width, height, n_len, n_class = 766, 80, 30, len(characters) + 1
from keras import backend as K
IMAGE_HEIGHT = 80
IMAGE_WIDTH = 766
# MAX_CAPTCHA = len(text)
MAX_CAPTCHA = 30
BATCH_SIZE = 64
EPOCH = 30

from keras.utils.np_utils import to_categorical

def text2vec(text):
    text_len = len(text)
    if text_len > MAX_CAPTCHA:
        raise ValueError('验证码最长30个字符')
    while text_len < MAX_CAPTCHA:
        text += '_'
        text_len += 1
    vector = np.zeros(n_len)

    def char2pos(c):
        if c == '_':
            k = 44
            return k
        for count, ch in enumerate(chinese_character):
            if c == ch:
                return 16 + count
        k = ord(c) - 48
        # if k > 9:
        #   k = ord(c) - 55
        #   if k > 35:
        #       k = ord(c) - 61
        #       if k > 61:
        #           raise ValueError('No Map')
        if c == '+':
            k = 10
        elif c == '-':
            k = 11
        elif c == '*':
            k = 12
        elif c == '(':
            k = 13
        elif c == ')':
            k = 14
        elif c == '!':
            k = 15
        # elif c == '=':
        #     k = 16
        elif k < 0:
            print 'No!', c
            quit()
        return k
    for i, c in enumerate(text):
        # idx = i * CHAR_SET_LEN + char2pos(c)
        vector[i] = char2pos(c)
    return vector

def convert2gray(img):
    if len(img.shape) > 2:
        gray = np.mean(img, -1)
        # 上面的转法较快，正规转法如下
        # r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
        # gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray
    else:
        return img


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

from keras.models import *
from keras.layers import *
rnn_size = 256

input_tensor = Input((width, height, 3))
x = input_tensor
for i in range(3):
    x = Convolution2D(filters=32 * 2 ** i, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = Convolution2D(filters=32 * 2 ** i, kernel_size=(3, 3), padding='same', activation='relu')(x)
    if i != 3:
        x = MaxPooling2D(pool_size=(2, 2))(x)

x = Convolution2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')(x)
x = Convolution2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')(x)



conv_shape = x.get_shape()
x = Reshape(target_shape=(int(conv_shape[1]), int(conv_shape[2]*conv_shape[3])))(x)

x = Dense(64, activation='relu')(x)

gru_1 = GRU(rnn_size, return_sequences=True, init='he_normal', name='gru1')(x)
gru_1b = GRU(rnn_size, return_sequences=True, go_backwards=True, init='he_normal', name='gru1_b')(x)
gru1_merged = merge([gru_1, gru_1b], mode='sum')

gru_2 = GRU(rnn_size, return_sequences=True, init='he_normal', name='gru2')(gru1_merged)
gru_2b = GRU(rnn_size, return_sequences=True, go_backwards=True, init='he_normal', name='gru2_b')(gru1_merged)
x = merge([gru_2, gru_2b], mode='concat')
x = Dropout(0.5)(x)
x = Dense(n_class, init='he_normal', activation='softmax')(x)
base_model = Model(input=input_tensor, output=x)

labels = Input(name='the_labels', shape=[n_len], dtype='float32')
input_length = Input(name='input_length', shape=[1], dtype='int64')
label_length = Input(name='label_length', shape=[1], dtype='int64')
loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([x, labels, input_length, label_length])

model = Model(input=[input_tensor, labels, input_length, label_length], output=[loss_out])
model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adadelta')

from keras.utils.vis_utils import plot_model
plot_model(model, to_file='/tmp/model.png', show_shapes=True)

# def gen(batch_size=128):
#     X = np.zeros((batch_size, width, height, 3), dtype=np.uint8)
#     y = np.zeros((batch_size, n_len), dtype=np.uint8)
#     while True:
#         generator = ImageCaptcha(width=width, height=height)
#         for i in range(batch_size):
#             random_str = ''.join([random.choice(characters) for j in range(4)])
#             X[i] = np.array(generator.generate_image(random_str)).transpose(1, 0, 2)
#             y[i] = [characters.find(x) for x in random_str]
#         yield [X, y, np.ones(batch_size)*int(conv_shape[1]-2), np.ones(batch_size)*n_len], np.ones(batch_size)
tmp_label = []
with open('/d_2t/image_contest_level_2/labels.txt') as f:
    lines = f.readlines()
    for line in lines:
        line = line.split(' ')[0]
        line = line.split(';')
        tmp_label.append(line)

def get_label(file_row, file_frac):
    return tmp_label[file_row][file_frac]

trick_tmp = []
with open('/d_2t/trick.txt') as f:
    lines = f.readlines()
    for line in lines:
        trick_tmp.append(line.decode('utf-8'))

def generate_val(seed=None):
    filenames = FILE_NAMES[125000:]
    while True:
        print 'Shuffling data'
        filenames = shuffle(filenames)
        for i in range(len(filenames) // BATCH_SIZE):
            train = np.ndarray((BATCH_SIZE, IMAGE_WIDTH, IMAGE_HEIGHT, 3))
            label = np.ndarray((BATCH_SIZE, n_len))
            for j in range(i * BATCH_SIZE, (i + 1) * BATCH_SIZE):
                img = (cv2.imread(filenames[j]))
                train[j - i * BATCH_SIZE, :, :, :] = img.transpose(1, 0, 2)

                filename = filenames[j]
                string = filename.split('/')[-1].split('.')[0].split('_')
                # print 'file_row', string[0], 'file_frac', string[1]
                file_row = int(string[0])
                if file_row < 100000:
                    file_frac = int(string[1])

                    label_re = get_label(file_row, file_frac).decode('utf-8')
                    label_re = label_re.replace('/', '!')
                    label[j - i * BATCH_SIZE] = text2vec(label_re)
                else:

                    label_re = trick_tmp[int(file_row) - 100000].replace('/', '!')[:-1]
                    label[j - i * BATCH_SIZE] = text2vec(label_re)

            yield [train, label, np.ones(BATCH_SIZE) * int(conv_shape[1] - 2), np.ones(BATCH_SIZE) * n_len], np.ones(
                BATCH_SIZE)


def generate_train(seed=None):
    filenames = FILE_NAMES[:125000]
    while True:
        print 'Shuffling data'
        filenames = shuffle(filenames)
        for i in range(len(filenames) // BATCH_SIZE):
            train = np.ndarray((BATCH_SIZE, IMAGE_WIDTH, IMAGE_HEIGHT, 3))
            label = np.ndarray((BATCH_SIZE, n_len))
            for j in range(i * BATCH_SIZE, (i + 1) * BATCH_SIZE):
                img = (cv2.imread(filenames[j]))
                train[j - i * BATCH_SIZE, :, :, :] = img.transpose(1, 0, 2)

                filename = filenames[j]
                string = filename.split('/')[-1].split('.')[0].split('_')
                # print 'file_row', string[0], 'file_frac', string[1]
                file_row = int(string[0])
                if file_row < 100000:
                    file_frac = int(string[1])

                    label_re = get_label(file_row, file_frac).decode('utf-8')
                    label_re = label_re.replace('/', '!')
                    label[j - i * BATCH_SIZE] = text2vec(label_re)
                else:
                    label_re = trick_tmp[int(file_row) - 100000].replace('/', '!')[:-1]
                    print label_re
                    print filename
                    quit()
                    label[j - i * BATCH_SIZE] = text2vec(label_re)

            yield [train, label, np.ones(BATCH_SIZE) * int(conv_shape[1] - 2), np.ones(BATCH_SIZE) * n_len], np.ones(
                BATCH_SIZE)


# def generate_val(seed=None):
#     tmp_label = []
#     with open('/d_2t/change_data/labels.txt') as f:
#         lines = f.readlines()
#         for line in lines:
#             line = line.split(' ')[0]
#             tmp_label.append(line)
#     # train = []
#     # label = []
#     # val_train = []
#     # val_label = []
#     filenames = FILE_NAMES[240000:]
#     while True:
#         print 'Shuffling data'
#         filenames = shuffle(filenames)
#         for i in range(len(filenames) // BATCH_SIZE):
#             train = np.ndarray((BATCH_SIZE, 180, 60, 1))
#             label = np.ndarray((BATCH_SIZE, n_len))
#             for j in range(i * BATCH_SIZE, (i + 1) * BATCH_SIZE):
#                 img = convert2gray(cv2.imread(filenames[j]))
#                 train[j - i * BATCH_SIZE, :, :, 0] = (img.transpose(1, 0) / 255)
#                 filename = filenames[j].split('/')[-1].split('.')[0]
#                 label[j - i * BATCH_SIZE] = (text2vec(tmp_label[int(filename)]))
#
#             yield [train, label, np.ones(BATCH_SIZE) * int(conv_shape[1] - 2), np.ones(BATCH_SIZE) * n_len], np.ones(
#                 BATCH_SIZE)


def evaluate(model, batch_num=20):
    batch_acc = 0
    generator = generate_val()
    for i in range(batch_num):
        [X_test, y_test, _, _], _  = next(generator)
        y_pred = base_model.predict(X_test)
        shape = y_pred[:,2:,:].shape
        out = K.get_value(K.ctc_decode(y_pred[:,2:,:], input_length=np.ones(shape[0])*shape[1])[0][0])[:, :30]
        # print out
        # print y_test
        if out.shape[1] == 30:
            batch_acc += ((y_test == out).sum(axis=1) == 30).mean()
    return batch_acc / batch_num


from keras.callbacks import *


class Evaluate(Callback):
    def __init__(self):
        self.accs = []

    def on_epoch_end(self, epoch, logs=None):
        acc = evaluate(base_model) * 100
        self.accs.append(acc)
        print
        print 'acc: %f%%' % acc



evaluator = Evaluate()
model.fit_generator(generate_train(), samples_per_epoch=125000 // BATCH_SIZE, nb_epoch=200,
                    callbacks=[EarlyStopping(patience=7), evaluator, ModelCheckpoint('/d_2t/lstm2_change_a_b.model',  save_best_only=True)],
                    validation_data=generate_val(), nb_val_samples=25000 // BATCH_SIZE)

model.fit_generator(generate_train(), samples_per_epoch=125000 // BATCH_SIZE, nb_epoch=200,
                    callbacks=[EarlyStopping(patience=10), evaluator, ModelCheckpoint('/d_2t/lstm2_change_a_b.model',  save_best_only=True)],
                    validation_data=generate_val(), nb_val_samples=25000 // BATCH_SIZE)

# model.fit_generator(gen(128), samples_per_epoch=51200 // 128, nb_epoch=200,
#                     callbacks=[EarlyStopping(patience=10), evaluator],
#                     validation_data=gen(128), nb_val_samples=1280 // 128)


# characters2 = characters + ' '
# [X_test, y_test, _, _], _  = next(gen(1))
# y_pred = base_model.predict(X_test)
# y_pred = y_pred[:,2:,:]
# out = K.get_value(K.ctc_decode(y_pred, input_length=np.ones(y_pred.shape[0])*y_pred.shape[1], )[0][0])[:, :4]
# out = ''.join([characters[x] for x in out[0]])
# y_true = ''.join([characters[x] for x in y_test[0]])
#
# plt.imshow(X_test[0].transpose(1, 0, 2))
# plt.title('pred:' + str(out) + '\ntrue: ' + str(y_true))
# argmax = np.argmax(y_pred, axis=2)[0]
# ansewr = list(zip(argmax, ''.join([characters2[x] for x in argmax])))
# plt.savefig('figure.png')
# with open('answer.pkl', 'w') as f:
#     pickle.dump(ansewr, f)


