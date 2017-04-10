from keras.layers import UpSampling2D, Lambda
from keras.models import Sequential

from model_component import add_seq_conv_block


def create_textnet(input_rows, input_cols, num_res_filters=128,
                   activation='relu', num_inner_blocks=5, batch_size=32):
    net = Sequential()
    add_seq_conv_block(net, num_res_filters // 4, 9, input_shape=(batch_size, input_rows, input_cols, 3),
                       activation=activation)
    add_seq_conv_block(net, num_res_filters // 2, 3, subsample=(2, 2), activation=activation)
    add_seq_conv_block(net, num_res_filters, 3, subsample=(2, 2), activation=activation)
    for i in range(num_inner_blocks):
        add_seq_conv_block(net, num_res_filters, 3, activation=activation)
        add_seq_conv_block(net, num_res_filters, 3, activation=activation)

    net.add(UpSampling2D())
    add_seq_conv_block(net, num_res_filters // 2, 3, activation=activation)
    net.add(UpSampling2D())
    add_seq_conv_block(net, num_res_filters // 4, 3, activation=activation)
    add_seq_conv_block(net, 3, 9, activation='tanh')
    net.add(Lambda(lambda x: x * 128.))
    return net
