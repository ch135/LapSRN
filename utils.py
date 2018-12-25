# -*- coding: utf-8 -*-
# @Time    : 2018/12/22 10:19
# @Author  : chenhao
# @FileName: utils
# @Software: PyCharm
import scipy.misc
import numpy as np
from PIL import Image
import math
from config import *
import matplotlib.pyplot as plt


def get_iamge(file_name):
    return scipy.misc.imread(file_name, mode="RGB")


def augment_imags(x):
    return x + 0.1 * x.std() * np.random.random(x.shape)


def nomalize_iamge(x):
    return x * (2. / 255.) - 1
    # return x*(1./255.)


def truncate_images(x):
    x = np.where(x > -1., x, -1)
    x = np.where(x < 1., x, 1.)
    return x


def img_psnr(img1, img2):
    img1 = np.array(Image.open(img1), dtype="float32")
    img2 = np.array(Image.open(img2), dtype="float")
    height = img1.shape[0]
    width = img1.shape[1]
    assert height == img2.shape[0] and width == img2.shape[1]
    R = img1[:, :, 0] - img2[:, :, 0]
    G = img1[:, :, 1] - img2[:, :, 1]
    B = img1[:, :, 2] - img2[:, :, 2]
    mser, mseg, mseb = R * R, G * G, B * B
    sum = mser.sum() + mseg.sum() + mseb.sum()
    mse = sum / (3 * height * width)
    psnr = 10 * math.log(255 * 255 / mse, 10)
    return psnr


def result_psnr():
    psnr_values = {}
    for i in range(250, 461):
        if 250 <= i < 280:
            path = config.train.train_path + "250_279/"
        elif 280 <= i < 461:
            path = config.train.train_path
        img1 = path + "train_sample_output.png"
        img2 = path + "train_predict_%s.png" % i
        psnr_values[i] = img_psnr(img1, img2)

    plt.figure()
    plt.xlim(250, 461)
    plt.ylim(33, 35)
    plt.xlabel("Epoch")
    plt.ylabel("PSNR(dB)")
    plt.plot(psnr_values.keys(), psnr_values.values())
    maxx = max(psnr_values, key=psnr_values.get)
    maxy = psnr_values[maxx]
    plt.plot([0, maxx], [maxy, maxy], "r--")
    plt.text(maxx - 25, maxy - 0.2, "(%s, %.2f)" % (maxx, maxy), fontdict={"color": "r"})
    plt.legend(loc="best", labels=["learning rate: 1)250~279: 0; 2)280~460:5 * 1.e-6; scale=4"])
    plt.savefig(config.model.result_path+"train/PSNR.png")
    plt.show()
