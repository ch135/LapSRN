# -*- coding: utf-8 -*-
# @Time    : 2018/12/22 10:19
# @Author  : chenhao
# @FileName: utils
# @Software: PyCharm
import scipy.misc
import numpy as np


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
