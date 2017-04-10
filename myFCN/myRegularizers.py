import glob
import six
import os
from keras.layers import concatenate, Conv2D, MaxPooling2D, Flatten, Dense
from keras import backend as K
from keras.applications import vgg16
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.engine import Input
from keras.engine import Model
from keras.layers.convolutional import Convolution2D, UpSampling2D
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.regularizers import Regularizer
from keras.optimizers import Adam
from keras.preprocessing.image import load_img, img_to_array
import numpy as np

import my_utils
import pascal_image_pair
import matplotlib.pylab as plt


# the gram matrix of an image tensor (feature-wise outer product)
# reshape output to change th format to tf format
def gram_matrix(x):
    assert K.ndim(x) == 4
    xs = K.shape(x)
    # features = K.reshape(x, (xs[0], xs[1], xs[2] * xs[3]))
    # gram = K.batch_dot(features, K.permute_dimensions(features, (0, 2, 1)))
    x = K.permute_dimensions(x, (0, 3, 1, 2))
    features = K.reshape(x, (xs[0], xs[1], xs[2] * xs[3]))
    gram = K.batch_dot(features, K.permute_dimensions(features, (0, 2, 1)))
    return gram


class FeatureStyleRegularizer(Regularizer):
    '''Gatys et al 2015 http://arxiv.org/pdf/1508.06576.pdf'''

    def __init__(self, target=None, weight=1.0, **kwargs):
        self.target = target
        self.weight = weight
        super(FeatureStyleRegularizer, self).__init__(**kwargs)

    def __call__(self, loss):
        output = self.layer.get_output(True)
        batch_size = K.shape(output)[0] // 2
        generated = output[:batch_size, :, :, :]
        loss += self.weight * K.mean(
            K.sum(K.square(gram_matrix(self.target) - gram_matrix(generated)), axis=(1, 2))
        ) / (4.0 * K.square(K.prod(K.shape(generated)[1:])))
        return loss


class FeatureContentRegularizer(Regularizer):
    '''Penalizes euclidean distance of content features.'''

    def __init__(self, weight=1.0, **kwargs):
        self.weight = weight
        super(FeatureContentRegularizer, self).__init__(**kwargs)

    def __call__(self, loss):
        output = self.layer.get_output(True)
        batch_size = K.shape(output)[0] // 2
        generated = output[:batch_size, :, :, :]
        content = output[batch_size:, :, :, :]
        loss += self.weight * K.mean(
            K.sum(K.square(content - generated), axis=(1, 2, 3))
        )
        return loss


# reshape output to change th format to tf format
class TVRegularizer(Regularizer):
    '''Enforces smoothness in image output.'''

    def __init__(self, weight=1.0, **kwargs):
        self.weight = weight
        super(TVRegularizer, self).__init__(**kwargs)

    def __call__(self, loss):
        x = self.layer.get_output(True)
        assert K.ndim(x) == 4
        # a = K.square(x[:, :, 1:, :-1] - x[:, :, :-1, :-1])
        # b = K.square(x[:, :, :-1, 1:] - x[:, :, :-1, :-1])
        a = K.square(x[:, 1:, :-1, :] - x[:, :-1, :-1, :])
        b = K.square(x[:, :-1, 1:, :] - x[:, :-1, :-1, :])
        loss += self.weight * K.mean(K.sum(K.pow(a + b, 1.25), axis=(1, 2, 3)))
        return loss


def add_seq_conv_block(net, filters, filter_size, activation='relu', subsample=(1, 1), input_shape=None):
    if input_shape:
        kwargs = dict(batch_input_shape=input_shape)
    else:
        kwargs = dict()
    net.add(Convolution2D(
        filters, (filter_size, filter_size), strides=subsample, padding='same', **kwargs))
    net.add(BatchNormalization())
    if isinstance(activation, six.string_types):
        if activation != 'linear':
            net.add(Activation(activation))
    else:
        net.add(activation())


def create_sequential_texture_net(input_rows, input_cols, num_res_filters=128,
                                  activation='relu', num_inner_blocks=5, batch_size=32):
    net = Sequential()
    # net.add(Input(batch_shape=))
    batch_shape = (batch_size, input_rows, input_cols, 3)
    add_seq_conv_block(net, num_res_filters // 4, 9, input_shape=batch_shape, activation=activation)
    add_seq_conv_block(net, num_res_filters // 2, 3, subsample=(2, 2), activation=activation)
    add_seq_conv_block(net, num_res_filters, 3, subsample=(2, 2), activation=activation)
    for i in range(num_inner_blocks):
        add_seq_conv_block(net, num_res_filters, 3, activation=activation)
        add_seq_conv_block(net, num_res_filters, 3, activation=activation)

    net.add(UpSampling2D())
    add_seq_conv_block(net, num_res_filters // 2, 3, activation=activation)
    net.add(UpSampling2D())
    add_seq_conv_block(net, num_res_filters // 4, 3, activation=activation)
    add_seq_conv_block(net, 3, 9, activation='linear')
    return net


def dumb_objective(x, y):
    '''Returns 0 in a way that makes everyone happy.

    Keras requires outputs and objectives but we're training purely upon the
    loss expressed by the regularizers.
    '''
    return 0.0 * y + 0.0 * x


if __name__ == '__main__':
    args = {
        'batch_size': 32,
        'max_height': 224,
        'max_width': 224,
        'activation': 'relu',
        'num_res_filters': 56,
        'num_blocks': 1
    }
    image_size = (args['max_height'], args['max_width'], 3)
    texnet = create_sequential_texture_net(args['max_height'], args['max_width'],
                                           activation=args['activation'], num_res_filters=args['num_res_filters'],
                                           num_inner_blocks=args['num_blocks'], batch_size=args['batch_size'])
    # texnet.summary()
    x1 = texnet.input
    x2 = Input(batch_shape=(args['batch_size'],) + image_size)

    y1 = texnet(x1)
    y2 = concatenate([y1, x2], axis=0)

    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1',
               activity_regularizer=FeatureStyleRegularizer)(y2)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1',
               activity_regularizer=FeatureStyleRegularizer    )(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1',
               activity_regularizer=FeatureStyleRegularizer)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1',
               activity_regularizer=FeatureStyleRegularizer)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2',
               activity_regularizer=FeatureContentRegularizer)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1',
               activity_regularizer=FeatureStyleRegularizer)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dense(1000, activation='softmax', name='predictions')(x)

    total_model = Model([x1, x2], x)
    total_model.summary()

    print(total_model.layers[-3:])


    # concante(x,y)
