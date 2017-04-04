# -*- coding: utf-8 -*-
import glob
import os

from keras.utils.data_utils import get_file

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
from keras.preprocessing import image
import numpy as np

image_path = "/home/andrew/VOC-CLS/"
default_size = 256
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
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,
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

    # model = vgg16.VGG16(include_top=False, weights='imagenet')
    # model = model(img_input)
    # x = Flatten()(model)
    # x = Dense(512, activation='relu', name='fc1')(x)
    # x = Dense(100, activation='relu', name='fc2')(x)
    # x = Dense(netout, activation='softmax', name='predictions')(x)
    #
    # model = Model([img_input], [x])
    # return model

    # Block 1
    x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv1')(img_input)
    x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)
    x = Dropout(0.25)(x)

    # Block 2
    x = Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='block2_conv1')(x)
    x = Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)
    x = Dropout(0.25)(x)

    # Block 3
    x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv1')(x)
    x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv2')(x)
    x = Convolution2D(256, 3, 3, border_mode='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)
    x = Dropout(0.25)(x)

    # Block 4
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv1')(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv2')(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)
    x = Dropout(0.25)(x)

    # Block 5
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv1')(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv2')(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    x = BatchNormalization(axis=3)(x)
    x = Dropout(0.25)(x)

    x = Flatten()(x)
    x = Dense(512, activation='relu', name='fc1')(x)
    x = Dense(100, activation='relu', name='fc2')(x)
    x = Dense(netout, activation='softmax', name='predictions')(x)

    inner_model = Model([img_input], [x])
    # TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
    # weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
    #                         TF_WEIGHTS_PATH_NO_TOP,
    #                         cache_subdir='models')
    # inner_model.load_weights(weights_path,by_name=True)
    return inner_model


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


def test_image(catergory):
    img = image.load_img(catergory, target_size=(256, 256))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


if __name__ == "__main__":

    out_model = build_model(nb_classes)
    train_generator = loadData(image_path)
    out_model.summary()
    out_model.compile(loss='categorical_crossentropy',
                      optimizer='sgd', metrics=['accuracy']
                      )

    newest = max(glob.iglob(weight_path + '*.hdf5'), key=os.path.getctime)
    out_model.load_weights(newest)
    print("Load newest:  ")
    print("       " + newest)
    # out_model.save("VOC-CLS.model")
    # exit()
    dir_path = image_path + "boat"
    category = 3
    output = []
    right = 0
    wrong = 0
    top = 0
    for parent, dirnames, filenames in os.walk(dir_path, topdown=False):
        print(filenames)
        for filename in filenames:
            img = test_image(parent + "/" + filename)
            y = out_model.predict(img)
            if np.argmax(y) == category:
                right += 1
            else:
                wrong += 1
                abc = np.argsort(-y)
                abc = abc[0]
                y=y[0]
                print(filename)
                print(abc[0:3])
                if category in abc[0:2]:
                    top+=1
                print(y[abc[0:3]])
    print("total:"+str(right+wrong))
    print("accuracy=" + str(right * 1. / (right + wrong)))
    print("top2=" + str((right+top) * 1. / (right + wrong)))
    exit()

    img = test_image('bicycle/2008_000615.jpg')
    y = out_model.predict(img)
    print("OK")
    print(np.sum(y), np.argmax(y))
    print(y)
    exit()

    # out_model.load_weights("weights.19-1.0897.hdf5")
    while True:
        try:
            newest = max(glob.iglob(weight_path + '*.hdf5'), key=os.path.getctime)
            out_model.load_weights(newest)
            print("Load newest:  ")
            print("       " + newest)


        finally:
            checkpointer = ModelCheckpoint(filepath=weight_path + '.{epoch:02d}-{val_loss:.4f}.hdf5',
                                           monitor='val_loss', verbose=1, save_best_only=True, period=10)
            # tensorboard = keras.callbacks.TensorBoard(log_dir="logs", histogram_freq=0)
            out_model.fit_generator(generator=train_generator, nb_epoch=1000, samples_per_epoch=12288,
                                    max_q_size=100, validation_data=train_generator,
                                    nb_val_samples=1000, callbacks=[checkpointer], verbose=1)

            # out_model = train_net(out_model, train_generator, weight_path="weigths")
            # out_model = train_net(out_model, train_generator, weight_path="weigths", finetune=True)
