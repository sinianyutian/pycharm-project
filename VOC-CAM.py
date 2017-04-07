import glob

import keras.backend as K
from keras.applications.imagenet_utils import preprocess_input
from keras.callbacks import ModelCheckpoint
from keras.engine import Input
from keras.engine import Model
from keras.models import Sequential, load_model
from keras.layers.core import Flatten, Dense
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.pooling import AveragePooling2D
from keras.layers.convolutional import ZeroPadding2D
import matplotlib.pylab as plt
import numpy as np
import theano.tensor.nnet.abstract_conv as absconv
import cv2
import h5py
import os
from keras.preprocessing import image

from keras.preprocessing.image import ImageDataGenerator
from keras.utils.data_utils import get_file


def preprocessing_img(img):
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    img = np.squeeze(img, axis=0)
    return img


def loadData(image_path):
    train_gen = ImageDataGenerator(
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,
        preprocessing_function=preprocessing_img)
    # rescale=1. / 255)

    inner_train_generator = train_gen.flow_from_directory(
        image_path,
        target_size=(256, 256),
        batch_size=32,
        class_mode='categorical')

    return inner_train_generator


def VGGCAM(nb_classes, num_input_channels=1024):
    """
    Build Convolution Neural Network

    args : nb_classes (int) number of classes

    returns : model (keras NN) the Neural Net model
    """

    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(224, 224, 3)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))

    # Add another conv layer with ReLU + GAP
    model.add(Convolution2D(num_input_channels, 3, 3, activation='relu', border_mode="same"))
    model.add(AveragePooling2D((14, 14)))
    model.add(Flatten())
    # Add the W layer
    model.add(Dense(nb_classes, activation='softmax'))

    model.name = "VGGCAM"

    return model


def get_classmap(model, X, nb_classes, batch_size, num_input_channels, ratio):
    inc = model.layers[0].input
    conv6 = model.layers[-4].output
    conv6_resized = absconv.bilinear_upsampling(conv6, ratio,
                                                batch_size=batch_size,
                                                num_input_channels=num_input_channels)
    WT = model.layers[-1].W.T
    conv6_resized = K.reshape(conv6_resized, (-1, num_input_channels, 256 * 256))
    classmap = K.dot(WT, conv6_resized).reshape((-1, nb_classes, 256, 256))
    get_cmap = K.function([inc], classmap)
    return get_cmap([X])


if __name__ == '__main__':
    model = VGGCAM(20)
    model.summary()
    model = load_model('VOC-CLS.model')
    x = model.input
    y = model.layers[-8].output
    # Add another conv layer with ReLU + GAP
    y = Convolution2D(1024, 3, 3, activation='relu', border_mode="same")(y)
    y = AveragePooling2D((16, 16))(y)
    y = Flatten()(y)
    # Add the W layer
    y = Dense(20, activation='softmax')(y)
    voc_cam = Model(x, y)
    voc_cam.summary()
    voc_cam.compile(loss='categorical_crossentropy',
                    optimizer='sgd', metrics=['accuracy']
                    )

    train_generator = loadData(image_path="/home/andrew/VOC-CLS/")
    weight_path = "voc_cam_"
    # checkpointer = ModelCheckpoint(filepath=weight_path + 'weights.{epoch:02d}-{loss:.4f}.hdf5',
    #                                monitor='loss', verbose=1, save_best_only=True, period=2)
    #
    # voc_cam.fit_generator(generator=train_generator, nb_epoch=100, samples_per_epoch=14000,
    #                       max_q_size=100, nb_val_samples=2000, callbacks=[checkpointer])

    newest = max(glob.iglob(weight_path + '*.hdf5'), key=os.path.getctime)
    voc_cam.load_weights(newest)
    print("Load newest:  ")
    print("       " + newest)

    image_path = "/home/andrew/VOC-CLS/"
    dir_path = image_path + "train"
    category = 0
    output = []
    right = 0
    wrong = 0
    top = 0


    def test_image(catergory):
        img = image.load_img(catergory, target_size=(256, 256))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        return x


    inc = voc_cam.layers[0].input
    conv6 = voc_cam.layers[-4].output
    model = Model(inc, conv6)
    model.summary()
    catergory=image_path+'train/2008_002452.jpg'
    label=18

    img = test_image(catergory)
    y = voc_cam.predict(img)
    abc = np.argsort(-y)
    abc = abc[0]
    y = y[0]
    print(catergory)
    print(abc[0:3])
    print(y[abc[0:3]])
    features = model.predict(img)
    features = np.squeeze(features)
    cam_weigths = K.get_value(K.transpose(voc_cam.layers[-1].W))

    # features = bilinear(features, 256, 256)
    features=np.transpose(np.reshape(features,(-1,1024)))
    print(features.shape)
    cam_image = np.dot(cam_weigths,features)
    print(cam_image.shape)
    cam_image=cam_image.reshape((20,16,16))
    im_ori = image.load_img(catergory, target_size=(256, 256))
    im_ori = np.squeeze(im_ori)
    print(im_ori.shape)
    plt.figure()
    plt.imshow(im_ori)
    # plt.figure()
    # plt.imshow(cam_image[label, :, :],
    #                       cmap="jet",
    #                       alpha=0.5,
    #                       interpolation='nearest',shape=(256,256))
    #
    # plt.show()

    def function(img, m, n):
        height, width, channels = img.shape
        emptyImage = np.zeros((m, n, channels))
        value = 0
        sh = m / height
        sw = n / width
        for i in range(m):
            for j in range(n):
                x = i / sh
                y = j / sw
                p = (i + 0.0) / sh - x
                q = (j + 0.0) / sw - y
                x = int(x) - 1
                y = int(y) - 1
                for k in range(1):
                    if x + 1 < m and y + 1 < n:
                        value = img[x, y] * (1 - p) * (1 - q) + img[x, y + 1] * q * (1 - p) + img[x + 1, y] * (
                            1 - q) * p + img[x + 1, y + 1] * p * q
                emptyImage[i, j] = value
        return emptyImage
    bi_img=cam_image[label,:,:].copy().reshape(16,16,1)
    bi_img=function(bi_img,256,256)
    bi_img=bi_img.squeeze()
    print(bi_img.shape)
    # plt.figure()
    plt.imshow(bi_img[ :, :],
                          cmap="jet",
                          alpha=0.5,
                          interpolation='nearest',shape=(256,256))

    plt.show()
    exit()
    # conv6_resized = absconv.bilinear_upsampling(conv6, ratio,
    #                                             batch_size=batch_size,
    #                                             num_input_channels=num_input_channels)
    # WT = model.layers[-1].W.T
    # conv6_resized = K.reshape(conv6_resized, (-1, num_input_channels, 256 * 256))
    # classmap = K.dot(WT, conv6_resized).reshape((-1, nb_classes, 256, 256))
    # get_cmap = K.function([inc], classmap)
    # return get_cmap([X])
    # for parent, dirnames, filenames in os.walk(dir_path, topdown=False):
    #     print(filenames)
    #     for filename in filenames:
    #         img = test_image(parent + "/" + filename)
    #         y = voc_cam.predict(img)
    #         if np.argmax(y) == category:
    #             right += 1
    #         else:
    #             wrong += 1
    #             abc = np.argsort(-y)
    #             abc = abc[0]
    #             y = y[0]
    #             print(filename)
    #             print(abc[0:3])
    #             if category in abc[0:2]:
    #                 top += 1
    #             print(y[abc[0:3]])
    # print("total:" + str(right + wrong))
    # print("accuracy=" + str(right * 1. / (right + wrong)))
    # print("top2=" + str((right + top) * 1. / (right + wrong)))
    # exit()

    # batch_size = 1
    # classmap = get_classmap(model,
    #                         im.reshape(1, 3, 256, 256),
    #                         nb_classes,
    #                         batch_size,
    #                         num_input_channels=num_input_channels,
    #                         ratio=ratio)
    #
    # plt.imshow(im_ori)
    # plt.imshow(classmap[0, label, :, :],
    #            cmap="jet",
    #            alpha=0.5,
    #            interpolation='nearest')
    # plt.show()
