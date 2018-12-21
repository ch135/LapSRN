# -*- coding: utf-8 -*-
# @Time    : 2018/12/21 16:09
# @Author  : chenhao
# @FileName: model.py
# @Software: PyCharm
import tensorflow as tf
import tensorlayer as tl
import config


def lrelu(x):
    return tf.maximum(x * 0.2, x)


def LapSRNSingleLevel(net_image, net_feature, reuse=False):
    with tf.variable_scope("Model_level", reuse=reuse):
        tl.layers.set_name_reuse(reuse)

        net_tmp = net_feature
        for d in range(config.model.reblock_depth):
            net_tmp = tl.layers.PReluLayer(net_tmp, name="prelu_D%s" % d)
            net_tmp = tl.layers.Conv2dLayer(net_tmp, shape=[3, 3, 64, 64], strides=[1, 1, 1, 1],
                                            W_init=tf.contrib.layers.xavier_initializer(), name="conv_D%s" % d)

        net_feature = tl.layers.ElementwiseLayer(prev_layer=[net_tmp, net_feature], combine_fn=tf.add,
                                                 name="add_feature")

        net_feature = tl.layers.PReluLayer(net_feature, name="preul_feature")
        net_feature = tl.layers.Conv2dLayer(net_feature, [3, 3, 64, 256], strides=[1, 1, 1, 1],
                                            W_init=tf.contrib.layers.xavier_initializer(),
                                            name="upconv_feature")
        net_feature = tl.layers.SubpixelConv2d(net_feature, scale=2, n_out_channel=3, act=lrelu,
                                               name="subpixl_feature")
        gradient_level = tl.layers.Conv2dLayer(net_feature, shape=[3, 3, 64, 3], strides=[1, 1, 1, 1],
                                               W_init=tf.contrib.layers.xavier_initializer(), name="grad")

        net_image = tl.layers.Conv2dLayer(net_image, shape=[3, 3, 3, 12], strides=[1, 1, 1, 1],
                                          W_init=tf.contrib.layers.xavier_initializer(), name="upconv_image")
        net_image = tl.layers.SubpixelConv2d(net_image, scale=2, n_out_channel=3,
                                             name="subpixl_iamge")
        net_image = tl.layers.ElementwiseLayer(layers=[gradient_level, net_image], combine_fn=tf.add,
                                               name="add_image")
    return net_image, net_feature, gradient_level


def LapSRN(inputs, is_train=False, reuse=False):
    with tf.variable_scope("LapSRN", reuse=reuse) as vs:
        tl.layers.set_name_reuse(reuse)

        inputs_lever = tl.layers.InputLayer(inputs, name="inputs_layer")
        net_feature = tl.layers.Conv2dLayer(inputs_lever, shape=[3, 3, 3, 64], strides=[1, 1, 1, 1],
                                            W_init=tf.contrib.layers.xavier_initializer(),
                                            name="init_cov")
        net_image = inputs_lever
        # 2X for each level
        net_image1, net_feature1, net_gradient1 = LapSRNSingleLevel(net_image, net_feature, reuse=reuse)
        net_image2, net_feature2, net_gradient2 = LapSRNSingleLevel(net_image1, net_feature1, reuse=True)

        return net_image2, net_gradient2, net_image1, net_gradient1
