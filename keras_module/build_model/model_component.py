import six

from keras.layers import Activation, Conv2D, Deconv2D, BatchNormalization, Dropout


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


def add_seq_conv_block(net, filters, filter_size, activation='relu', subsample=(1, 1), input_shape=None):
    if input_shape:
        kwargs = dict(batch_input_shape=input_shape)
    else:
        kwargs = dict()
    net.add(Conv2D(
        filters, (filter_size, filter_size), strides=subsample, padding='same', **kwargs))
    net.add(BatchNormalization())
    if isinstance(activation, six.string_types):
        if activation != 'linear':
            net.add(Activation(activation))
    else:
        net.add(activation())
