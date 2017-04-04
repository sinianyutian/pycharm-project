###set backend and image_idm
import os

from keras.datasets import cifar10
from keras.engine import Input
from keras.engine import Model
from keras.engine import merge
from keras.layers import AveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils

os.environ['KERAS_BACKEND'] = 'tensorflow'
from keras import backend as K

K.set_image_dim_ordering('tf')
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import Adam, SGD
from keras.layers.normalization import BatchNormalization
import numpy as np
import sys
from os.path import isfile, join
import os.path
import glob

n = 8
batch_size = 128
nb_classes = 10
nb_epoch = 64000
data_augmentation = True
weight_path = 'cifar'

# input image dimensions
img_rows, img_cols = 32, 32
# The CIFAR10 images are RGB.
img_channels = 3

# The data, shuffled and split between train and test sets:
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

if K.image_dim_ordering() == 'tf':
    bn_axis = 3
else:
    bn_axis = 1


def identity_block(input_tensor, kernel_size, nb_filter, stage, block, subsumpling=False):
    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + '_' + block + '_branch'
    bn_name_base = 'bn' + str(stage) + '_' + block + '_branch'

    if (subsumpling):
        x = Convolution2D(nb_filter, kernel_size, kernel_size, border_mode='same',
                          subsample=(2, 2), name=conv_name_base + '2a')(input_tensor)
    else:
        x = Convolution2D(nb_filter, kernel_size, kernel_size, border_mode='same',
                          name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter, kernel_size, kernel_size,
                      border_mode='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    if (subsumpling):
        x1 = Convolution2D(nb_filter, 1, 1, border_mode='same', subsample=(2, 2),
                           name=conv_name_base + '2c')(input_tensor)
        x = merge([x, x1], mode='sum')
    else:
        x = merge([x, input_tensor], mode='sum')
    x = Activation('relu')(x)
    return x


input_shape = X_train.shape[1:]
img_input = Input(shape=input_shape)

x = Convolution2D(16, 3, 3, border_mode='same', name='conv1')(img_input)
x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
x = Activation('relu')(x)

for i in range(n):
    x = identity_block(x, 3, 16, stage=2, block=str(i))

for i in range(n):
    if (i == 0):
        x = identity_block(x, 3, 32, stage=3, block=str(i), subsumpling=True)
    else:
        x = identity_block(x, 3, 32, stage=3, block=str(i))

for i in range(n):
    if (i == 0):
        x = identity_block(x, 3, 64, stage=4, block=str(i), subsumpling=True)
    else:
        x = identity_block(x, 3, 64, stage=4, block=str(i))

x = AveragePooling2D((2, 2), name='avg_pool')(x)
x = Flatten()(x)
x = Dense(nb_classes, activation='softmax', name='fc1000')(x)

model = Model(img_input, x)
model.summary()

datagen = ImageDataGenerator(
    featurewise_center=True,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False)  # randomly flip images

# Compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied).
datagen.fit(X_train)

checkpointer = ModelCheckpoint(filepath=weight_path + 'weights.{epoch:02d}-{val_loss:.4f}.hdf5',
                               monitor='val_loss', verbose=1, save_best_only=False, period=10)
tensorboard = TensorBoard(log_dir="logs", histogram_freq=0)

lr = 1.0
i = 1
for nb_epoch in [32000, 16000, 18000]:
    lr /= 10
    model.compile(loss='mean_squared_error', optimizer=SGD(lr=lr, momentum=0.9, decay=0.00001),
                  metrics=['accuracy'])

    if (i > 0):
        newest = max(glob.iglob(weight_path + '*.hdf5'), key=os.path.getctime)
        model.load_weights(newest)
    i += 1

    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(datagen.flow(X_train, Y_train,
                                     batch_size=batch_size),
                        samples_per_epoch=X_train.shape[0],
                        nb_epoch=nb_epoch,
                        validation_data=(X_test, Y_test),
                        callbacks=[checkpointer, tensorboard], verbose=2)
