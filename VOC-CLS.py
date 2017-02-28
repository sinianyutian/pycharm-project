# -*- coding: utf-8 -*-
import glob
import os

os.environ['KERAS_BACKEND'] = 'tensorflow'
from keras import backend as K

K.set_image_dim_ordering('tf')

import keras
from keras.applications import vgg16, resnet50
from keras.applications.imagenet_utils import preprocess_input
from keras.callbacks import ModelCheckpoint
from keras.engine import Input
from keras.engine import Model
from keras.layers import Flatten, Dense, Convolution2D, MaxPooling2D, Dropout, Activation, BatchNormalization
# from keras import backend as K
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

image_path = "/home/andrew/VOC-CLS/"
default_size = 224
nb_classes = 20
LR = 0.01
weight_path = "weigths"


def preprocessing_img(img):
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    img = np.squeeze(img, axis=0)
    return img


def loadData(image_path):
    train_gen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True,
        preprocessing_function=preprocessing_img)
    # rescale=1. / 255)

    inner_train_generator = train_gen.flow_from_directory(
        image_path,
        target_size=(default_size, default_size),
        batch_size=32,
        class_mode='categorical')

    return inner_train_generator


def build_model(netout):
    if K.image_dim_ordering() == 'th':
        default_shape = (3, default_size, default_size)
    else:
        default_shape = (default_size, default_size, 3)

    img_input = Input(default_shape)

    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1

    model = vgg16.VGG16(include_top=False, weights='imagenet')
    model.trainable = False
    model = model(img_input)
    x = Flatten()(model)
    x = Dense(512, activation='relu', name='fc1')(x)
    x = Dense(100, activation='relu', name='fc2')(x)
    x = Dense(netout, activation='softmax', name='predictions')(x)

    model = Model([img_input], [x])
    return model

    # model = Sequential()
    #
    # model.add(Convolution2D(32, 3, 3, border_mode='same',
    #                         input_shape=default_shape))
    # model.add(Activation('relu'))
    # model.add(Convolution2D(32, 3, 3))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))
    #
    # model.add(Convolution2D(64, 3, 3, border_mode='same'))
    # model.add(Activation('relu'))
    # model.add(Convolution2D(64, 3, 3))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))
    #
    # model.add(Flatten())
    # model.add(Dense(512))
    # model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(netout))
    # model.add(Activation('softmax'))

    # return model

    # # Block 1
    # x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv1')(img_input)
    # x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv2')(x)
    # x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    # x = BatchNormalization(axis=bn_axis)(x)
    # x = Activation('relu')(x)
    # x = Dropout(0.25)(x)
    #
    # # Block 2
    # x = Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='block2_conv1')(x)
    # x = Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='block2_conv2')(x)
    # x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
    # x = BatchNormalization(axis=bn_axis)(x)
    # x = Activation('relu')(x)
    # x = Dropout(0.25)(x)
    #
    # # Block 3
    # x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv1')(x)
    # x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv2')(x)
    # x = Convolution2D(256, 3, 3, border_mode='same', name='block3_conv3')(x)
    # x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    # x = BatchNormalization(axis=bn_axis)(x)
    # x = Activation('relu')(x)
    # x = Dropout(0.25)(x)
    #
    # # Block 4
    # x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv1')(x)
    # x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv2')(x)
    # x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv3')(x)
    # x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
    # x = BatchNormalization(axis=bn_axis)(x)
    # x = Activation('relu')(x)
    # x = Dropout(0.25)(x)

    # # Block 5
    # x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv1')(x)
    # x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv2')(x)
    # x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv3')(x)
    # x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    # x = BatchNormalization(axis=3)(x)
    # x = Dropout(0.25)(x)

    # x = Flatten()(x)
    # x = Dense(512, activation='relu', name='fc1')(x)
    # x = Dense(100, activation='relu', name='fc2')(x)
    # x = Dense(netout, activation='softmax', name='predictions')(x)
    #
    # inner_model = Model([img_input], [x])
    # return inner_model


def train_net(model, train_generator, weight_path, finetune=False):
    if finetune:
        newest = max(glob.iglob(weight_path + '*.hdf5'), key=os.path.getctime)
        model.load_weights(newest)
        lr = LR / 10
    else:
        lr = LR
    # adam = Adam(lr=lr)
    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd', metrics=['accuracy']
                  )
    checkpointer = ModelCheckpoint(filepath=weight_path + 'weights.{epoch:02d}-{val_loss:.4f}.hdf5',
                                   monitor='val_loss', verbose=1, save_best_only=True)
    tensorboard = keras.callbacks.TensorBoard(log_dir="logs", histogram_freq=0)

    model.fit_generator(generator=train_generator, nb_epoch=100, samples_per_epoch=13200,
                        max_q_size=100, validation_data=train_generator,
                        nb_val_samples=3000, callbacks=[checkpointer, tensorboard])

    newest = max(glob.iglob(weight_path + '*.hdf5'), key=os.path.getctime)
    model.load_weights(newest)
    return model


if __name__ == "__main__":

    out_model = build_model(nb_classes)
    train_generator = loadData(image_path)
    out_model.summary()
    out_model.compile(loss='categorical_crossentropy',
                      optimizer='sgd', metrics=['accuracy']
                      )
    while True:
        try:
            newest = max(glob.iglob(weight_path + '*.hdf5'), key=os.path.getctime)
            out_model.load_weights(newest)
        finally:
            checkpointer = ModelCheckpoint(filepath=weight_path + 'weights.{epoch:02d}-{val_loss:.4f}.hdf5',
                                       monitor='val_loss', verbose=1, save_best_only=True, period=100)
            # tensorboard = keras.callbacks.TensorBoard(log_dir="logs", histogram_freq=0)
            out_model.fit_generator(generator=train_generator, nb_epoch=1000, samples_per_epoch=12288,
                                max_q_size=100, validation_data=train_generator,
                                nb_val_samples=1000, callbacks=[checkpointer], verbose=1)

        # out_model = train_net(out_model, train_generator, weight_path="weigths")
        # out_model = train_net(out_model, train_generator, weight_path="weigths", finetune=True)
