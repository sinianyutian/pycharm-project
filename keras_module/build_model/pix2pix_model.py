from keras.engine import Input
from keras.engine import Model
from keras.layers import Activation, Conv2D, \
    LeakyReLU, concatenate
from model_component import conv_bn_relu

# To keep size,the size shoule satisfy input_size % 128 ==0
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


# To keep size,the size shoule satisfy input_size % 32 ==0
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
