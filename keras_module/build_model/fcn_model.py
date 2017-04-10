from keras.applications import vgg16
from keras.engine import Input
from keras.engine import Model
from keras.layers import Convolution2D, BatchNormalization, Activation, Dropout, Conv2DTranspose, merge
from keras.optimizers import Adam
from ..hwh_loss import fcn32_acc,fcn32_loss


def fcn32_model():
    vgg_model = vgg16.VGG16(include_top=False, weights='imagenet')

    x = Input((224, 224, 3))
    y = vgg_model(x)
    y = Convolution2D(4096, (7, 7), activation='relu', border_mode='same', name='fc6')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = Dropout(0.25)(y)
    y = Convolution2D(4096, (1, 1), activation='relu', border_mode='same', name='fc7')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = Dropout(0.25)(y)
    y = Convolution2D(21, (1, 1), activation='relu', border_mode='same', name='scor_fr')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = Dropout(0.25)(y)
    y = Conv2DTranspose(21, (32, 32), strides=(32, 32))(y)
    fcn32 = Model(x, y)
    adam = Adam()
    fcn32.compile(optimizer=adam, loss=fcn32_loss, metrics=[fcn32_acc])
    return fcn32


def facade_model(img_size, channel_out):
    vgg_model = vgg16.VGG16(include_top=False, weights='imagenet')

    x = Input((img_size, img_size, 3))
    y = vgg_model(x)
    y = Convolution2D(4096, (7, 7), activation='relu', border_mode='same', name='fc6')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = Dropout(0.25)(y)
    y = Convolution2D(4096, (1, 1), activation='relu', border_mode='same', name='fc7')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = Dropout(0.25)(y)
    y = Convolution2D(21, (1, 1), activation='relu', border_mode='same', name='scor_fr')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = Dropout(0.25)(y)
    y = Conv2DTranspose(channel_out, (32, 32), strides=(32, 32))(y)
    facade = Model(x, y)
    adam = Adam()
    facade.compile(optimizer=adam, loss=fcn32_loss, metrics=[fcn32_acc])
    return facade


def fcn16_model():
    x = Input((224, 224, 3))
    vgg_model = vgg16.VGG16(include_top=False, weights='imagenet', input_tensor=x)
    pool4 = vgg_model.get_layer('block4_pool')

    pool4_y = pool4.output
    y = vgg_model.output

    y = Convolution2D(4096, (7, 7), activation='relu', border_mode='same', name='fc6')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = Dropout(0.25)(y)
    y = Convolution2D(4096, (2, 2), activation='relu', name='fc7')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = Dropout(0.25)(y)
    y = Convolution2D(21, (1, 1), activation='relu', border_mode='same', name='scor_fr')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = Dropout(0.25)(y)
    y = Conv2DTranspose(21, (4, 4), strides=(2, 2))(y)

    y = merge([y, pool4_y], mode='concat')

    y = Conv2DTranspose(21, (32, 32), strides=(16, 16), padding='same')(y)

    fcn16 = Model(x, y)
    fcn16.summary()
    adam = Adam()
    fcn16.compile(optimizer=adam, loss=fcn32_loss, metrics=[fcn32_acc])
    return fcn16