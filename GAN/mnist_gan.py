#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train an Auxiliary Classifier Generative Adversarial Network (ACGAN) on the
MNIST dataset. See https://arxiv.org/abs/1610.09585 for more details.

You should start to see reasonable images after ~5 epochs, and good images
by ~15 epochs. You should use a GPU, as the convolution-heavy operations are
very slow on the CPU. Prefer the TensorFlow backend if you plan on iterating, as
the compilation time can be a blocker using Theano.

Timings:

Hardware           | Backend | Time / Epoch
-------------------------------------------
 CPU               | TF      | 3 hrs
 Titan X (maxwell) | TF      | 4 min
 Titan X (maxwell) | TH      | 7 min

Consult https://github.com/lukedeo/keras-acgan for more information and
example output
"""
from __future__ import print_function

from collections import defaultdict

try:
    import cPickle as pickle
except ImportError:
    import pickle
from PIL import Image
from gan_model import cifar_gan, cifar_dis,generator_model,discriminator_model
import os

os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras.backend as K

K.set_image_dim_ordering('th')

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Embedding, merge, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Convolution2D
from keras.models import Sequential, Model
from keras.optimizers import Adam, SGD
from keras.utils.generic_utils import Progbar
import numpy as np

np.random.seed(1337)


def build_generator(latent_size):
    # we will map a pair of (z, L), where z is a latent vector and L is a
    # label drawn from P_c, to image space (..., 1, 28, 28)
    cnn = Sequential()

    cnn.add(Dense(1024, input_dim=latent_size, activation='relu'))
    cnn.add(Dense(128 * 7 * 7, activation='relu'))
    cnn.add(Reshape((128, 7, 7)))

    # upsample to (..., 14, 14)
    cnn.add(UpSampling2D(size=(2, 2)))
    cnn.add(Convolution2D(256, 5, 5, border_mode='same',
                          activation='relu', init='glorot_normal'))

    # upsample to (..., 28, 28)
    cnn.add(UpSampling2D(size=(2, 2)))
    cnn.add(Convolution2D(128, 5, 5, border_mode='same',
                          activation='relu', init='glorot_normal'))

    # take a channel axis reduction
    cnn.add(Convolution2D(1, 2, 2, border_mode='same',
                          activation='tanh', init='glorot_normal'))

    return cnn


def build_discriminator():
    # build a relatively standard conv net, with LeakyReLUs as suggested in
    # the reference paper
    cnn = Sequential()

    cnn.add(Convolution2D(32, 3, 3, border_mode='same', subsample=(2, 2),
                          input_shape=(1, 28, 28)))
    cnn.add(LeakyReLU())
    cnn.add(Dropout(0.3))

    cnn.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1)))
    cnn.add(LeakyReLU())
    cnn.add(Dropout(0.3))

    cnn.add(Convolution2D(128, 3, 3, border_mode='same', subsample=(2, 2)))
    cnn.add(LeakyReLU())
    cnn.add(Dropout(0.3))

    cnn.add(Convolution2D(256, 3, 3, border_mode='same', subsample=(1, 1)))
    cnn.add(LeakyReLU())
    cnn.add(Dropout(0.3))

    cnn.add(Flatten())

    image = Input(shape=(1, 28, 28))

    features = cnn(image)

    # first output (name=generation) is whether or not the discriminator
    # thinks the image that is being shown is fake, and the second output
    # (name=auxiliary) is the class that the discriminator thinks the image
    # belongs to.
    fake = Dense(1, activation='sigmoid', name='generation')(features)

    return Model(input=image, output=fake)


if __name__ == '__main__':

    # batch and latent size taken from the paper
    nb_epochs = 1000
    batch_size = 128
    latent_size = 100


    sgd=SGD(lr=0.01, momentum=0.9)
    # build the discriminator
    discriminator = discriminator_model()
    discriminator.compile(
        optimizer=sgd,
        loss=['binary_crossentropy']
    )

    # build the generator
    # generator = build_generator(latent_size)
    generator = generator_model(latent_size)
    generator.compile(optimizer=sgd,
                      loss='binary_crossentropy')

    latent = Input(shape=(latent_size,))

    # get a fake image
    fake = generator(latent)

    # we only want to be able to train generation for the combined model
    discriminator.trainable = False
    fake = discriminator(fake)
    combined = Model(input=latent, output=fake)

    combined.compile(
        optimizer=sgd,
        loss=['binary_crossentropy']
    )

    # get our mnist data, and force it to be of shape (..., 1, 28, 28) with
    # range [-1, 1]
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    X_train = np.expand_dims(X_train, axis=1)

    X_test = (X_test.astype(np.float32) - 127.5) / 127.5
    X_test = np.expand_dims(X_test, axis=1)

    nb_train, nb_test = X_train.shape[0], X_test.shape[0]

    progress_bar = Progbar(target=nb_epochs)
    for epoch in range(nb_epochs):
        print('Epoch {} of {}'.format(epoch + 1, nb_epochs))

        progress_bar.update(epoch)
        # generate a new batch of noise
        noise = np.random.uniform(-1, 1, (nb_train, latent_size))
        generated_images = generator.predict(noise, verbose=0)
        # print(generated_images.shape)
        # print(X_train.shape)
        X = np.concatenate((X_train, generated_images))
        y = np.array([1] * nb_train + [0] * nb_train)
        discriminator.fit(X, y, shuffle=True, validation_split=0.1, batch_size=batch_size, nb_epoch=1)

        noise = np.random.uniform(-1, 1, (nb_train, latent_size))
        trick = np.ones(nb_train)
        combined.fit(noise, trick, batch_size=batch_size, shuffle=True, validation_split=0.1, nb_epoch=1)
        # generator.save_weights(
        #     '../model-hdf5/gan_params_generator_epoch_{0:03d}.hdf5'.format(epoch), True)
        # discriminator.save_weights(
        #     '../model-hdf5/gan_params_discriminator_epoch_{0:03d}.hdf5'.format(epoch), True)

        # generate some digits to display
        noise = np.random.uniform(-1, 1, (100, latent_size))

        # get a batch to display
        generated_images = generator.predict(
            noise, verbose=0)

        # arrange them into a grid
        img = (np.concatenate([r.reshape(-1, 28)
                               for r in np.split(generated_images, 10)
                               ], axis=-1) * 127.5 + 127.5).astype(np.uint8)

        Image.fromarray(img).save(
            'gan_plot_epoch_{0:03d}_generated.png'.format(epoch))
