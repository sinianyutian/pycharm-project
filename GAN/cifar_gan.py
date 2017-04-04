#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

from collections import defaultdict
from keras.datasets import cifar10

import gan_model
from keras.engine import Input

try:
    import cPickle as pickle
except ImportError:
    import pickle
from PIL import Image

from keras.models import Sequential, Model
from keras.optimizers import Adam, SGD
from keras.utils.generic_utils import Progbar
import numpy as np

np.random.seed(1337)

if __name__ == '__main__':

    # batch and latent size taken from the paper
    nb_epochs = 1000
    batch_size = 64
    latent_size = 100
    adam = Adam(lr=0.0002, beta_1=0.5)
    # build the discriminator
    tmp_model = gan_model.discriminator_model()
    tmp_aaa = Input((32, 32, 3))
    tmp_bbb = tmp_model(tmp_aaa)
    discriminator = Model(tmp_aaa, tmp_bbb)
    discriminator.compile(
        optimizer=adam,
        loss=['binary_crossentropy']
    )
    discriminator.summary()

    # build the generator
    generator = gan_model.generator_model(inputdim=latent_size,xdim=2,ydim=2)
    generator.compile(optimizer=adam,
                      loss='binary_crossentropy')

    latent = Input(shape=(latent_size,))

    # get a fake image
    fake = generator(latent)

    # we only want to be able to train generation for the combined model
    discriminator.trainable = False
    fake = discriminator(fake)
    combined = Model(input=latent, output=fake)

    combined.compile(
        optimizer=adam,
        loss=['binary_crossentropy']
    )
    # combined.summary()
    # exit()
    # get our mnist data, and force it to be of shape (..., 1, 28, 28) with
    # range [-1, 1]
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    # X_train = X_train.astype(np.float32)
    # x_mean = np.mean(X_train)
    # x_std = np.mean(X_test)
    # X_train -= x_mean
    # X_train /= x_std
    # X_train = np.expand_dims(X_train, axis=1)

    X_test = (X_test.astype(np.float32) - 127.5) / 127.5
    # X_test = X_test.astype(np.float32)
    # X_test -= x_mean
    # X_test /= x_std
    # X_test = np.expand_dims(X_test, axis=1)

    nb_train, nb_test = X_train.shape[0], X_test.shape[0]


    # see if the discriminator can figure itself out...
    # print(X_test.shape)
    # exit()

    # img = (np.concatenate([r.reshape(-1, 32, 3)
    #                        for r in np.split(X_train[0:100], 10)
    #                        ], axis=1) * 127.5 + 127.5).astype(np.uint8)
    #
    # Image.fromarray(img).save(
    #     'cifar_gan_plot_epoch_{0:03d}_generated.png'.format(111))
    # exit()
    train_history = defaultdict(list)
    test_history = defaultdict(list)

    for epoch in range(nb_epochs):
        print('Epoch {} of {}'.format(epoch + 1, nb_epochs))

        nb_batches = int(X_train.shape[0] / batch_size)
        progress_bar = Progbar(target=nb_batches)

        epoch_gen_loss = []
        epoch_disc_loss = []

        for index in range(nb_batches):
            progress_bar.update(index)
            # generate a new batch of noise
            noise = np.random.uniform(-1, 1, (batch_size, latent_size))

            # get a batch of real images
            image_batch = X_train[index * batch_size:(index + 1) * batch_size] * 1.0
            label_batch = y_train[index * batch_size:(index + 1) * batch_size] * 1.0

            # generate a batch of fake images, using the generated labels as a
            # conditioner. We reshape the sampled labels to be
            # (batch_size, 1) so that we can feed them into the embedding
            # layer as a length one sequence
            generated_images = generator.predict(
                noise, verbose=0)

            X = np.concatenate((image_batch, generated_images))
            y = np.array([1] * batch_size + [0] * batch_size)

            # see if the discriminator can figure itself out...
            epoch_disc_loss.append(discriminator.train_on_batch(X, y))
            # make new noise. we generate 2 * batch size here such that we have
            # the generator optimize over an identical number of images as the
            # discriminator
            noise = np.random.uniform(-1, 1, (2 * batch_size, latent_size))

            # we want to train the genrator to trick the discriminator
            # For the generator, we want all the {fake, not-fake} labels to say
            # not-fake
            trick = np.ones(2 * batch_size)

            epoch_gen_loss.append(combined.train_on_batch(
                noise, trick))

        print('\nTesting for epoch {}:'.format(epoch + 1))

        # evaluate the testing loss here

        # generate a new batch of noise
        noise = np.random.uniform(-1, 1, (nb_test, latent_size))

        generated_images = generator.predict(
            noise, verbose=False)

        X = np.concatenate((X_test, generated_images))
        y = np.array([1] * nb_test + [0] * nb_test)

        # see if the discriminator can figure itself out...
        discriminator_test_loss = discriminator.evaluate(
            X, y, verbose=False)

        discriminator_train_loss = np.mean(np.array(epoch_disc_loss), axis=0)

        # make new noise
        noise = np.random.uniform(-1, 1, (2 * nb_test, latent_size))

        trick = np.ones(2 * nb_test)

        generator_test_loss = combined.evaluate(
            noise,
            trick, verbose=False)

        generator_train_loss = np.mean(np.array(epoch_gen_loss), axis=0)

        # generate an epoch report on performance
        train_history['generator'].append(generator_train_loss)
        train_history['discriminator'].append(discriminator_train_loss)

        test_history['generator'].append(generator_test_loss)
        test_history['discriminator'].append(discriminator_test_loss)

        print('{0:<22s} | {1:<22s}'.format(
            'component', discriminator.metrics_names[-1]))
        print('-' * 65)

        ROW_FMT = '{0:<22s} | {1:4f}'
        print(ROW_FMT.format('generator (train)',
                             train_history['generator'][-1]))
        print(ROW_FMT.format('generator (test)',
                             test_history['generator'][-1]))
        print(ROW_FMT.format('discriminator (train)',
                             train_history['discriminator'][-1]))
        print(ROW_FMT.format('discriminator (test)',
                             test_history['discriminator'][-1]))

        # save weights every epoch
        if epoch > 10 and epoch % 10 == 0:
            generator.save_weights(
                'cifar_gan_{0:03d}.hdf5'.format(epoch), True)
            discriminator.save_weights(
                'cifar_dis_{0:03d}.hdf5'.format(epoch), True)

        # generate some digits to display
        noise = np.random.uniform(-1, 1, (100, latent_size))

        # get a batch to display
        generated_images = generator.predict(
            noise, verbose=0)

        # arrange them into a grid
        img = (np.concatenate([r.reshape(-1, 32, 3)
                               for r in np.split(generated_images, 10)
                               ], axis=1) * 127.5 + 127.5).astype(np.uint8)
        # img = (np.concatenate([r.reshape(-1, 32, 3)
        #                        for r in np.split(generated_images, 10)
        #                        ], axis=1) * x_mean + x_std).astype(np.uint8)
        Image.fromarray(img).save(
            'cifar_gan_plot_epoch_{0:03d}_generated.png'.format(epoch))
