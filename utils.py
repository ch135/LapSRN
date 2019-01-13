# -*- coding: utf-8 -*-
# @Time    : 2018/12/22 10:19
# @Author  : chenhao
# @FileName: utils
# @Software: PyCharm
import scipy.misc
import numpy as np
import math
from config import *
import matplotlib.pyplot as plt
import cv2 as cv

#  等同于 matlab的 imread(input_filename)
def get_iamge(file_name):
    return scipy.misc.imread(file_name, mode="RGB")


def augment_imags(x):
    return x + 0.1 * x.std() * np.random.random(x.shape)


def modcrop(img):
    scale = config.model.scale
    if (len(img.shape) == 3):
        width = img.shape[0] - img.shape[0] % scale
        height = img.shape[1] - img.shape[1] % scale
        img = img[1:width, 1:height, :]
    return img

# 等同于 matlab 的 im2double(img)
def nomalize_iamge(x):
    # return x * (2. / 255.) - 1
    return x*(1./255.)


def truncate_images(x):
    x = np.where(x > -1., x, -1)
    x = np.where(x < 1., x, 1.)
    return x


def img_psnr(img1, img2):
    scale = config.model.scale
    img1 = np.array(scipy.misc.imread(img1, mode="YCbCr"), dtype="float32")
    img2 = np.array(scipy.misc.imread(img2, mode="YCbCr"), dtype="float32")
    height = img1.shape[0]
    width = img1.shape[1]
    img1 = img1[scale + 1:height - scale, scale + 1:width - scale, :]
    img2 = img2[scale + 1:height - scale, scale + 1:width - scale, :]
    assert img1.shape[0] == img2.shape[0] and img1.shape[1] == img2.shape[1]
    Y = img1[:, :, 0] - img2[:, :, 0]
    Cb = img1[:, :, 1] - img2[:, :, 1]
    Cr = img1[:, :, 2] - img2[:, :, 2]
    mser, mseg, mseb = Y * Y, Cb * Cb, Cr * Cr
    sum = mser.sum() + mseg.sum() + mseb.sum()
    mse = sum / (3 * height * width)
    psnr = 10 * math.log(255 * 255 / mse, 10)
    return float("{:2.3f}".format(psnr))


def train_psnr(epoch):
    psnr_values = {}
    for i in range(epoch):
        path = config.train.train_path
        img1 = path + "train_sample_output.png"
        img2 = path + "train_predict_%s.png" % i
        psnr_values[i] = img_psnr(img1, img2)
    plt.figure()
    plt.xlim(0, epoch)
    plt.ylim(15, 40)
    yticket=np.arange(15, 40, 5)
    plt.yticks(yticket, yticket)
    plt.xlabel("Epoch")
    plt.ylabel("PSNR(dB)")
    plt.plot(psnr_values.keys(), psnr_values.values())
    maxx = max(psnr_values, key=psnr_values.get)
    maxy = psnr_values[maxx]
    plt.plot([0, maxx, maxx, maxx], [maxy, maxy, 0, maxy], "r--", "r--")
    plt.text(maxx - 45, maxy + 0.5, "(%s, %.2f)" % (maxx, maxy), fontdict={"color": "r"})
    # plt.legend(loc="best", labels=["learning rate:0.000005 to 0.000003 to 0.000001 to 0.000001"])
    plt.savefig(config.model.result_path + "train/PSNR_%d.png" % epoch)
    plt.show()


# 双立方插值
def image_bicubic(file, path):
    img = cv.imread(file)
    dst = cv.resize(img, dsize=None, fx=0.25, fy=0.25, interpolation=cv.INTER_CUBIC)
    cv.imwrite(path, dst)


if __name__ == '__main__':
     train_psnr(150)