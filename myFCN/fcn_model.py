import keras.backend as K
from keras.applications import vgg16
from keras.engine import Input
from keras.engine import Model
from keras.layers import Convolution2D, Dropout, BatchNormalization, Activation, Conv2DTranspose, merge, Conv2D, \
    Deconv2D, LeakyReLU, concatenate
from keras.optimizers import Adam


def fcn32_loss(y_true, y_pred):
    return K.categorical_crossentropy(y_pred, y_true, from_logits=True)


def fcn32_acc(y_true, y_pred):
    return K.mean(K.cast(K.equal(K.argmax(y_true, axis=-1),
                                 K.argmax(y_pred, axis=-1)),
                         K.floatx()))


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


def conv_bn_relu(x, channel, bn=True, sample='down',
                 activation=Activation('relu'), dropout=False):
    if sample == 'down':
        y = Conv2D(channel, (4, 4), strides=(2, 2), padding='same')(x)
    else:
        y = Deconv2D(channel, (4, 4), strides=(2, 2), activation='relu', padding='same')(x)
    if bn:
        y = BatchNormalization()(y)
    if dropout:
        y = Dropout(0.2)(y)
    y = activation(y)

    return y


def pix2pix_generator(nb_in, nb_out):
    x = Input((256, 256, nb_in))
    # encoder
    y = conv_bn_relu(x, 64, bn=True, sample='down', activation=LeakyReLU(), dropout=False)

    y = conv_bn_relu(y, 128, bn=True, sample='down', activation=LeakyReLU(), dropout=False)
    y2 = y
    y = conv_bn_relu(y, 256, bn=True, sample='down', activation=LeakyReLU(), dropout=False)
    y3 = y
    y = conv_bn_relu(y, 512, bn=True, sample='down', activation=LeakyReLU(), dropout=False)
    y4 = y
    y = conv_bn_relu(y, 512, bn=True, sample='down', activation=LeakyReLU(), dropout=False)
    y5 = y
    y = conv_bn_relu(y, 512, bn=True, sample='down', activation=LeakyReLU(), dropout=False)
    y6 = y
    y = conv_bn_relu(y, 512, bn=True, sample='down', activation=LeakyReLU(), dropout=False)

    # decoder
    y = conv_bn_relu(y, 512, bn=True, sample='up', activation=Activation('relu'), dropout=True)
    y = concatenate([y, y6])
    y = conv_bn_relu(y, 512, bn=True, sample='up', activation=Activation('relu'), dropout=True)
    y = concatenate([y, y5])
    y = conv_bn_relu(y, 512, bn=True, sample='up', activation=Activation('relu'), dropout=True)
    y = concatenate([y, y4])
    y = conv_bn_relu(y, 256, bn=True, sample='up', activation=Activation('relu'), dropout=False)
    y = concatenate([y, y3])
    y = conv_bn_relu(y, 128, bn=True, sample='up', activation=Activation('relu'), dropout=False)
    y = concatenate([y, y2])
    y = conv_bn_relu(y, 64, bn=True, sample='up', activation=Activation('relu'), dropout=False)

    y = conv_bn_relu(y, nb_out, bn=True, sample='up', activation=LeakyReLU(), dropout=False)

    model = Model(x, y)
    # model.summary()
    return model


def pix2pix_generator_single(nb_in, nb_out, input_size=(256, 256)):
    x = Input(input_size + (nb_in,))
    # encoder
    y = conv_bn_relu(x, 64, bn=True, sample='down', activation=LeakyReLU(), dropout=False)

    y = conv_bn_relu(y, 128, bn=True, sample='down', activation=LeakyReLU(), dropout=False)
    y2 = y
    y = conv_bn_relu(y, 256, bn=True, sample='down', activation=LeakyReLU(), dropout=False)
    y3 = y
    y = conv_bn_relu(y, 512, bn=True, sample='down', activation=LeakyReLU(), dropout=False)
    y4 = y
    y = conv_bn_relu(y, 512, bn=True, sample='down', activation=LeakyReLU(), dropout=False)
    y5 = y
    y = conv_bn_relu(y, 512, bn=True, sample='down', activation=LeakyReLU(), dropout=False)

    # decoder
    y = conv_bn_relu(y, 512, bn=True, sample='up', activation=Activation('relu'), dropout=True)
    y = concatenate([y, y5])
    y = conv_bn_relu(y, 512, bn=True, sample='up', activation=Activation('relu'), dropout=True)
    y = concatenate([y, y4])
    y = conv_bn_relu(y, 256, bn=True, sample='up', activation=Activation('relu'), dropout=False)
    y = concatenate([y, y3])
    y = conv_bn_relu(y, 128, bn=True, sample='up', activation=Activation('relu'), dropout=False)
    y = concatenate([y, y2])
    y = conv_bn_relu(y, 64, bn=True, sample='up', activation=Activation('relu'), dropout=False)

    y = conv_bn_relu(y, nb_out, bn=True, sample='up', activation=Activation('sigmoid'), dropout=False)

    model = Model(x, y)
    # model.summary()
    return model


def pix2pix_discriminator(nb_in, nb_out):
    x1 = Input((256, 256, nb_in))
    x2 = Input((256, 256, nb_out))
    # encoder
    y1 = conv_bn_relu(x1, 32, bn=False, sample='down', activation=LeakyReLU(), dropout=False)
    y2 = conv_bn_relu(x2, 32, bn=False, sample='down', activation=LeakyReLU(), dropout=False)
    y = concatenate([y1, y2])
    y = conv_bn_relu(y, 128, bn=False, sample='down', activation=LeakyReLU(), dropout=False)
    y = conv_bn_relu(y, 256, bn=False, sample='down', activation=LeakyReLU(), dropout=False)
    y = Conv2D(1, (3, 3), strides=(1, 1))(y)
    model = Model([x1, x2], y)
    # model = Model(x, y)
    # model.summary()
    return model


def pix2pix_discriminator_single(nb_in, input_size=(256, 156)):
    x = Input(input_size + (nb_in,))
    # encoder
    y = conv_bn_relu(x, 32, bn=False, sample='down', activation=LeakyReLU(), dropout=False)
    y = conv_bn_relu(y, 128, bn=False, sample='down', activation=LeakyReLU(), dropout=False)
    y = conv_bn_relu(y, 256, bn=False, sample='down', activation=LeakyReLU(), dropout=False)
    y = Conv2D(1, (3, 3), strides=(1, 1))(y)
    model = Model(x, y)
    # model = Model(x, y)
    # model.summary()
    return model


if __name__ == '__main__':
    # fcn16_model()
    # pix2pix_generator()
    # pix2pix_discriminator(3,4)
    model = pix2pix_generator_single(3, 21, input_size=(192, 192))
    # model = pix2pix_discriminator_single(3,input_size=(224,224))
    model.summary()
