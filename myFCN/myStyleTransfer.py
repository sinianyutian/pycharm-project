import glob
import matplotlib.pylab as plt
import os
import six

import numpy as np
from keras import backend as K
from keras.applications import vgg16
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.engine import Input
from keras.engine import Model
from keras.layers import Lambda
from keras.layers.convolutional import Convolution2D, UpSampling2D
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing.image import load_img, img_to_array

import my_utils
import pascal_image_pair
from keras_module.data_draw.draw_pascal import draw_images_pair
from keras_module.hwh_callbacks import DrawEpoch
# from keras_module.hwh_layers import BatchRenormalization
from keras_module.hwh_layers import BatchRenormalization, InstanceNormalization
from myVgg16 import myVGG16

model = Sequential()
args = {
    'batch_size': 32,
    'max_height': 168,
    'max_width': 168,
    'activation': 'relu',
    'num_res_filters': 56,
    'num_blocks': 1
}

img_nrows, img_ncols, channels = args['max_height'], args['max_width'], 3

weigths = {'loss_weight': 0.025,
           'stlye_weigth': 1.0,
           'tv_weigth': 1.0}

# style_path = "../style_image/style/asheville.jpg"
style_path = "../style_image/style/brushstrokes.jpg"

weight_path = 'style-transfer'
nb_epochs = 10000
batch_size = 32
data_type = 'trainval'
is_test = True

pasacl_train = pascal_image_pair.PascalVOC2012SegmentationDatasetImagePair(data_type,
                                                                           target_size=(img_nrows, img_ncols))
data_total_len = len(pasacl_train)
my_gen_train = my_utils.ImageIterator(pasacl_train, batch_size)

vgg16_model = myVGG16()
vgg16_model.load_weigth()


# x= Input((224,224,3))
# y=model(x)
# y=Convolution2D(3,3)(y)
# y_model = Model(x,y)
#
# layers=model.get_layer(index=7)
# layers.trainable=True
# # y_model.compile(Adam(),loss='binary_crossentropy')
# print(y_model.summary())
# # print(layers)
# exit()


def preprocess_image(image_path):
    img = load_img(image_path, target_size=(args['max_height'], args['max_width']))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg16.preprocess_input(img)
    return img


def deprocess_image(x):
    # Remove zero-center by mean pixel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x


# def draw_images_pair(img1_datas, img2_datas, batch_size=5, is_save=True, prefix='st-', index_pro=1):
#     plt.figure(figsize=(100, 40))
#     for index in range(batch_size):
#         datum = img1_datas[index]
#         datum = deprocess_image(datum)
#         label = img2_datas[index]
#         label = deprocess_image(label)
#         plt.subplot(2, batch_size, index + 1)
#         plt.imshow(datum)
#         plt.subplot(2, batch_size, batch_size + index + 1)
#         plt.imshow(label)
#     if is_save:
#         plt.savefig(prefix + str(index_pro) + '.jpg')
#     else:
#         plt.show()
#     return


#
# for _ in range(1):
#     x_test, y_test = my_gen_train.next()
#     print(x_test.shape)
#     print(np.max(x_test),np.min(x_test))
#     draw_images_pair(x_test, y_test, index_pro=111, is_save=False)
# exit()

style_image = preprocess_image(style_path)
# print(style_image.shape)
style_image = np.repeat(style_image, axis=0, repeats=batch_size)


# print(style_image.shape)
# style_image = deprocess_image(style_image[4, :, :, :])
# plt.imshow(style_image)
# plt.show()
# exit()


def add_seq_conv_block(net, filters, filter_size, activation='relu', subsample=(1, 1), input_shape=None, is_IB=True):
    if input_shape:
        kwargs = dict(batch_input_shape=input_shape)
    else:
        kwargs = dict()
    net.add(Convolution2D(
        filters, (filter_size, filter_size), strides=subsample, padding='same', **kwargs))
    if is_IB:
        net.add(InstanceNormalization())
    else:
        net.add(BatchNormalization())
    if isinstance(activation, six.string_types):
        if activation != 'linear':
            net.add(Activation(activation))
    else:
        net.add(activation())


def create_sequential_texture_net(input_rows, input_cols, num_res_filters=128,
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


def gram_matrix(x):
    assert K.ndim(x) == 4
    xs = K.shape(x)
    # features = K.reshape(x, (xs[0], xs[1], xs[2] * xs[3]))
    # gram = K.batch_dot(features, K.permute_dimensions(features, (0, 2, 1)))
    x = K.permute_dimensions(x, (0, 3, 1, 2))
    # print(K.get_variable_shape(x))
    features = K.reshape(x, (xs[0], xs[1], xs[2] * xs[3]))
    gram = K.batch_dot(features, K.permute_dimensions(features, (0, 2, 1)))
    # print(K.get_variable_shape(gram))
    return gram


def style_loss(style, combination):
    assert K.ndim(style) == 4
    assert K.ndim(combination) == 4
    target = style
    generated = combination
    var_shape = K.get_variable_shape(style)
    var_squar_prod = np.square(np.prod(var_shape[1:]))
    # print(var_squar_prod)
    return K.mean(
        K.sum(K.square(gram_matrix(target) - gram_matrix(generated)), axis=(1, 2))
    ) / (4.0 * var_squar_prod)


def tv_loss(x):
    assert K.ndim(x) == 4
    # a = K.square(x[:, :, 1:, :-1] - x[:, :, :-1, :-1])
    # b = K.square(x[:, :, :-1, 1:] - x[:, :, :-1, :-1])
    a = K.square(x[:, 1:, :-1, :] - x[:, :-1, :-1, :])
    b = K.square(x[:, :-1, 1:, :] - x[:, :-1, :-1, :])
    return K.mean(K.sum(K.pow(a + b, 1.25), axis=(1, 2, 3)))


# y_true:content_image y_pred:generated image
def style_total_loss(y_true, y_pred):
    loss = K.variable(0.)
    style_reference_image = K.variable(style_image)
    input_tensor = K.concatenate([y_pred, y_true,
                                  style_reference_image], axis=0)
    # print(K.get_variable_shape(y_true))
    # print(K.get_variable_shape(input_tensor))
    # print(K.get_variable_shape(input_tensor))
    content, style_layers = vgg16_model.get_layer(input_tensor)

    generated = content[:batch_size, :, :, :]
    contented = content[batch_size:2 * batch_size, :, :, :]
    # print(K.get_variable_shape(generated))
    # print(K.get_variable_shape(contented))
    # exit()

    loss += weigths['loss_weight'] * K.mean(
        K.sum(K.square(contented - generated), axis=(1, 2, 3))
    )

    for style_layer in style_layers:
        combination_features = style_layer[:batch_size, :, :, :]
        style_reference_features = style_layer[2 * batch_size:3 * batch_size, :, :, :]
        sl = style_loss(style_reference_features, combination_features)
        loss += weigths['stlye_weigth'] * sl

    loss += weigths['tv_weigth'] * tv_loss(y_pred[:batch_size, :, :, :])

    return loss


if __name__ == '__main__':
    texnet = create_sequential_texture_net(args['max_height'], args['max_width'],
                                           activation=args['activation'], num_res_filters=args['num_res_filters'],
                                           num_inner_blocks=args['num_blocks'], batch_size=args['batch_size'])
    #
    # texnet.summary()
    # exit()

    texnet.compile(Adam(), loss=style_total_loss)
    is_test = False
    nb_epochs = 100
    img_abc, label_abc = my_gen_train.next()
    img_abc, label_abc = my_gen_train.next()
    imb_abc = img_abc.copy()
    label_abc = label_abc.copy()

    # Train
    while True:
        try:
            newest = max(glob.iglob(weight_path + '*.hdf5'), key=os.path.getctime)
            texnet.load_weights(newest)
            print("Load newest:  ")
            print("       " + newest)
        finally:
            if is_test:
                # texnet.load_weights('style-transfer-epoch.014.hdf5')

                for indexabc in range(4):
                    img_test, label_test = my_gen_train.next()
                    # print(np.max(img_test), np.min(img_test))
                    label_pred = texnet.predict(img_test)

                    # print(np.max(label_pred), np.min(label_pred))
                    # exit()
                    draw_images_pair(img_test, label_pred,
                                     batch_size=5, is_save=False, prefix='st-', index_pro=indexabc,is_block=True)
                exit()

            # checkpointer = ModelCheckpoint(filepath=weight_path + '-epoch.{epoch:03d}-{loss:.4f}.hdf5',
            #                                monitor='loss', verbose=1, period=2)

            checkpointer = ModelCheckpoint(filepath=weight_path + '-epoch.{epoch:03d}.hdf5',
                                           verbose=1, period=5)
            # plt.ion()
            draw_function = lambda x, y, epoch: draw_images_pair(x, y, epoch,
                                                                 batch_size=5, is_save=True, prefix='st-')
            drawepoch = DrawEpoch(img_abc, texnet, draw_function=draw_function, period=2)
            # tensorboard = TensorBoard(log_dir="logs", histogram_freq=0)
            texnet.fit_generator(generator=my_gen_train, steps_per_epoch=int(data_total_len / batch_size),
                                 epochs=nb_epochs, max_q_size=10,
                                 callbacks=[checkpointer, drawepoch], verbose=1, workers=2)
            exit()
