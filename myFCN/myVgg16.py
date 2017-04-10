# -*- coding: utf-8 -*-
"""VGG16 model for Keras.

# Reference

- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)

"""
from __future__ import absolute_import
from __future__ import print_function

from keras.engine import Input
from keras.engine import Model
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils.data_utils import get_file

WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'


def get_conv(channel, name):
    return Conv2D(channel, (3, 3), activation='relu', padding='same', name=name)


def get_pool(name):
    return MaxPooling2D((2, 2), strides=(2, 2), name=name)


class myVGG16(object):
    def __init__(self):
        self.block1_conv1 = get_conv(64, 'block1_conv1')
        self.block1_conv2 = get_conv(64, 'block1_conv2')
        self.block1_pool = get_pool('block1_pool')

        self.block2_conv1 = get_conv(128, 'block2_conv1')
        self.block2_conv2 = get_conv(128, 'block2_conv2')
        self.block2_pool = get_pool('block2_pool')

        self.block3_conv1 = get_conv(256, 'block3_conv1')
        self.block3_conv2 = get_conv(256, 'block3_conv2')
        self.block3_conv3 = get_conv(256, 'block3_conv3')
        self.block3_pool = get_pool('block3_pool')

        self.block4_conv1 = get_conv(512, 'block4_conv1')
        self.block4_conv2 = get_conv(512, 'block4_conv2')
        self.block4_conv3 = get_conv(512, 'block4_conv3')
        self.block4_pool = get_pool('block4_pool')

        self.block5_conv1 = get_conv(512, 'block5_conv1')
        self.block5_conv2 = get_conv(512, 'block5_conv2')
        self.block5_conv3 = get_conv(512, 'block5_conv3')
        self.block5_pool = get_pool('block5_pool')

        # self.block5_conv1.trainable=False

    def create_model(self, input):
        x = self.block1_conv1(input)
        x = self.block1_conv2(x)
        x = self.block1_pool(x)

        x = self.block2_conv1(x)
        x = self.block2_conv2(x)
        x = self.block2_pool(x)

        x = self.block3_conv1(x)
        x = self.block3_conv2(x)
        x = self.block3_conv3(x)
        x = self.block3_pool(x)

        x = self.block4_conv1(x)
        x = self.block4_conv2(x)
        x = self.block4_conv3(x)
        x = self.block4_pool(x)

        x = self.block5_conv1(x)
        x = self.block5_conv2(x)
        x = self.block5_conv3(x)
        x = self.block5_pool(x)
        return x

    def load_weigth(self, input_size=(224, 224, 3), trainable=False):
        x = Input(input_size)
        y = self.create_model(x)
        model = Model(x, y)
        weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                WEIGHTS_PATH_NO_TOP,
                                cache_subdir='models')
        model.load_weights(weights_path)
        for layer in model.layers:
            layer.trainable = trainable
        return model

    def get_layer(self, input):
        x = self.block1_conv1(input)
        l1 = x
        x = self.block1_conv2(x)
        x = self.block1_pool(x)

        x = self.block2_conv1(x)
        l2 = x
        x = self.block2_conv2(x)
        x = self.block2_pool(x)

        x = self.block3_conv1(x)
        l3 = x
        x = self.block3_conv2(x)
        c = x
        x = self.block3_conv3(x)
        x = self.block3_pool(x)

        x = self.block4_conv1(x)
        l4 = x
        x = self.block4_conv2(x)
        x = self.block4_conv3(x)
        x = self.block4_pool(x)

        x = self.block5_conv1(x)
        l5 = x
        x = self.block5_conv2(x)
        x = self.block5_conv3(x)
        x = self.block5_pool(x)
        return c, [l1, l2, l3, l4, l5]
