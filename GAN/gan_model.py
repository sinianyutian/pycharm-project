import keras.backend as K

K.set_image_dim_ordering('th')
from keras.layers import Dense, Reshape, UpSampling2D, Convolution2D
from keras.layers import BatchNormalization, Activation, LeakyReLU, Dropout, Flatten
from keras.models import Sequential


def cifar_gan(input_size=100, ngf=32):
    cnn = Sequential()

    # size (ngf*8,4,4)
    cnn.add(Dense(ngf * 8, input_dim=input_size, activation='relu', init='glorot_uniform'))
    cnn.add(Dense(ngf * 8 * 4 * 4))
    # cnn.add(Reshape((4, 4, ngf * 8)))
    cnn.add(Reshape((ngf * 8, 4, 4)))
    cnn.add(BatchNormalization(mode=2))
    # cnn.add(LeakyReLU(0.2))
    cnn.add(Activation('relu'))

    # size (ngf*4,8,8)
    cnn.add(UpSampling2D(size=(2, 2)))
    cnn.add(Convolution2D(ngf * 4, 4, 4, border_mode='same', init='glorot_uniform',
                          bias=False))
    cnn.add(BatchNormalization(mode=2))
    # cnn.add(LeakyReLU(0.2))
    cnn.add(Activation('relu'))

    # size (ngf*2,16,16)
    cnn.add(UpSampling2D(size=(2, 2)))
    cnn.add(Convolution2D(ngf * 2, 2, 2, border_mode='same', init='glorot_uniform',
                          bias=False))
    cnn.add(BatchNormalization(mode=2))
    # cnn.add(LeakyReLU(0.2))
    cnn.add(Activation('relu'))

    # size (ngf*1,32,32)
    cnn.add(UpSampling2D(size=(2, 2)))
    cnn.add(Convolution2D(ngf * 1, 2, 2, border_mode='same', init='glorot_uniform',
                          bias=False))
    cnn.add(BatchNormalization())
    # cnn.add(LeakyReLU(0.2))
    cnn.add(Activation('relu'))

    # size (3,32,32)
    cnn.add(Convolution2D(3, 1, 1, border_mode='same',
                          bias=False))
    cnn.add(BatchNormalization())
    cnn.add(Activation('tanh'))

    return cnn


def cifar_dis(ndf=64):
    cnn = Sequential()
    # size ndf*32*32
    cnn.add(Convolution2D(ndf, 4, 4, border_mode='same',
                          bias=False,
                          input_shape=(3, 32, 32)))
    cnn.add(LeakyReLU(0.2))
    cnn.add(Dropout(0.3))

    # size ndf*2,16*16
    cnn.add(Convolution2D(ndf * 2, 4, 4, border_mode='same',
                          bias=False,
                          subsample=(2, 2)))
    cnn.add(BatchNormalization())
    cnn.add(LeakyReLU(0.2))
    cnn.add(Dropout(0.2))
    # size ndf*4,8*8
    cnn.add(Convolution2D(ndf * 4, 4, 4, border_mode='same',
                          bias=False,
                          subsample=(2, 2)))
    cnn.add(BatchNormalization())
    cnn.add(LeakyReLU(0.2))
    cnn.add(Dropout(0.2))
    # size ndf*8,4*4
    cnn.add(Convolution2D(ndf * 8, 4, 4, border_mode='same',
                          bias=False,
                          subsample=(2, 2)))
    cnn.add(BatchNormalization())
    cnn.add(LeakyReLU(0.2))
    cnn.add(Dropout(0.2))
    # cnn.add(Convolution2D(2, 4, 4,
    #                       bias=False,
    #                       subsample=(2, 2)))
    cnn.add(Flatten())
    cnn.add(Dense(256))
    cnn.add(LeakyReLU(0.2))
    cnn.add(Dropout(0.2))
    cnn.add(Dense(2, activation='softmax'))
    # cnn.add(Activation('sigmoid'))

    return cnn


def generator_model(inputdim=100, xdim=2, ydim=2):
    # xdim = 2, ydim = 2 results in prediction shape of (1, 3, 32, 32)
    # xdim = 4, ydim = 4 results in prediction shape of (1, 3, 64, 64)
    model = Sequential()
    model.add(Dense(input_dim=inputdim, output_dim=1024 * xdim * ydim))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Reshape((1024,xdim, ydim), input_shape=(inputdim,)))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(512, 5, 5, border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(256, 5, 5, border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(128, 5, 5,border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(1, 5, 5))
    model.add(Activation('tanh'))
    return model


def discriminator_model():
    model = Sequential()
    model.add(Convolution2D(128, 5, 5, subsample=(2, 2), input_shape=(1,28, 28), border_mode='same',activation='relu'))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.2))
    model.add(Convolution2D(256, 5, 5, subsample=(2, 2), border_mode='same',activation='relu'))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.2))
    model.add(Convolution2D(512, 5, 5, subsample=(2, 2), border_mode='same',activation='relu'))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.2))
    model.add(Convolution2D(1024, 5, 5, subsample=(2, 2), border_mode='same',activation='relu'))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(2, activation='softmax'))
    # model.add(Dense(output_dim=1))
    # model.add(Activation('sigmoid'))
    return model


if __name__ == '__main__':
    generator_model().summary()
    discriminator_model().summary()
    # cifar_dis().summary()
    # cifar_gan().summary()
