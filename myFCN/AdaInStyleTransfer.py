import glob
import os

import numpy as np
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam

from keras_module.build_model.myVgg16 import myVGG16
from keras_module.build_model.vggConv import build_vggConv
from keras_module.data_gen.data_interator import ImageIterator
from keras_module.data_gen.pascal_gen import Content_Style_Pair
from keras_module.hwh_callbacks import DrawEpoch
from keras_module.hwh_loss import content_loss, tv_loss, gram_matrix
import matplotlib.pylab as plt

from keras_module.image_utils import deprocess_image

style_dir = '/home/andrew/PycharmProjects/pycharm-project/style_image/style'
content_type = 'trainval'
# 32X
image_height = 160
image_width = 160
image_size = (image_height, image_width, 3)

batch_size = 32
nb_epochs = 400

weight_path = 'ada-style-transfer'
is_test = False

weigths = {'loss_weight': 0.0025,
           'stlye_weigth': 1.0,
           'tv_weigth': 1.0}

# vgg16 fromat
content_style_iter = Content_Style_Pair(style_dir, data_type=content_type,
                                        target_size=(image_height, image_width))
# for abc in content_style_iter.style_files:
#     print(abc)
# exit()

data_total_len = len(content_style_iter)
style_total_len = content_style_iter.style_len()
my_gen_train = ImageIterator(content_style_iter, batch_size, is_list=True)

adam = Adam()

vgg16_model = myVGG16()
vgg_model = vgg16_model.load_weigth()
# vgg_model.summary()

vggConv_model = build_vggConv(input_shape=image_size, my_vgg=vgg_model)
vggConv_model.summary()
# exit()

def ada_content_loss(y_true, y_pred):
    input_tensor = K.concatenate([y_pred, y_true], axis=0)
    content, _ = vgg16_model.get_layer(input_tensor)
    generated = content[:batch_size, :, :, :]
    contented = content[batch_size:2 * batch_size, :, :, :]

    return content_loss(contented, generated)


def style_loss_imp(style, combination, channal_num):
    assert K.ndim(style) == 4
    assert K.ndim(combination) == 4
    target = style
    generated = combination
    var_shape = K.get_variable_shape(style)
    var_squar_prod = np.square(np.prod([channal_num, image_width, image_height]))
    # var_squar_prod = np.square(np.prod(var_shape[1:]))
    # print(var_squar_prod)
    return K.mean(
        K.sum(K.square(gram_matrix(target) - gram_matrix(generated)), axis=(1, 2))
    ) / (4.0 * var_squar_prod)


def ada_style_loss(y_true, y_pred):
    loss = K.variable(0.0)
    input_tensor = K.concatenate([y_pred, y_true], axis=0)
    _, style_layers = vgg16_model.get_layer(input_tensor)
    layer_channel = [64, 128, 256, 512, 512]
    for index, style_layer in enumerate(style_layers):
        combination_features = style_layer[:batch_size, :, :, :]
        style_reference_features = style_layer[batch_size:2 * batch_size, :, :, :]
        sl = style_loss_imp(style_reference_features, combination_features, layer_channel[index])
        loss += sl
    return loss


def ada_tv_loss(y_true, y_pred):
    return K.variable(0.)
    # return tv_loss(y_pred)


def ada_draw_images_pair(img1_datas, img2_datas, index_pro=1, batch_size=5, is_save=True, prefix='st-', is_block=False):
    plt.figure(figsize=(100, 40))
    for index in range(batch_size):
        datum = img1_datas[0][index].copy()
        datum = deprocess_image(datum)
        label = img1_datas[1][index].copy()
        label = deprocess_image(label)
        output = img2_datas[2][index].copy()
        output = deprocess_image(output)
        plt.subplot(3, batch_size, index + 1)
        plt.imshow(datum)
        plt.subplot(3, batch_size, batch_size + index + 1)
        plt.imshow(label)
        plt.subplot(3, batch_size, 2 * batch_size + index + 1)
        plt.imshow(output)

    if is_save:
        plt.savefig(prefix + str(index_pro) + '.jpg')
    else:
        plt.show(block=is_block)
    return

#
# for _ in range(1):
#     img_test, label_test = next(my_gen_train)
#
#     ada_draw_images_pair(img_test, label_test,
#                          batch_size=5, is_save=False, prefix='st-', index_pro=123, is_block=True)
# exit()

vggConv_model.compile(adam, loss=[ada_content_loss, ada_style_loss, ada_tv_loss],
                      loss_weights=[weigths['loss_weight'], weigths['stlye_weigth'], weigths['tv_weigth']])

img_abc, _ = next(my_gen_train)

is_test=True
# Train
while True:
    try:
        newest = max(glob.iglob(weight_path + '*.hdf5'), key=os.path.getctime)
        vggConv_model.load_weights(newest)
        print("Load newest:  ")
        print("       " + newest)
    finally:
        if is_test:
            # texnet.load_weights('style-transfer-epoch.014.hdf5')
            for indexabc in range(4):
                img_test, label_test = my_gen_train.next()
                label_pred = vggConv_model.predict(img_test)
                ada_draw_images_pair(img_test, label_pred,
                                 batch_size=5, is_save=False, prefix='st-', index_pro=indexabc, is_block=True)
            exit()

        # checkpointer = ModelCheckpoint(filepath=weight_path + '-epoch.{epoch:03d}-{loss:.4f}.hdf5',
        #                                monitor='loss', verbose=1, period=2)

        checkpointer = ModelCheckpoint(filepath=weight_path + '-epoch.{epoch:03d}.hdf5',
                                       verbose=1, period=20)
        draw_function = lambda x, y, epoch: ada_draw_images_pair(x, y, epoch,
                                                                 batch_size=5, is_save=True, prefix='st-')
        drawepoch = DrawEpoch(img_abc, vggConv_model, draw_function=draw_function, period=10)
        # tensorboard = TensorBoard(log_dir="logs", histogram_freq=0)
        vggConv_model.fit_generator(generator=my_gen_train, steps_per_epoch=int(data_total_len / batch_size),
                                    epochs=nb_epochs, max_q_size=10,
                                    callbacks=[checkpointer, drawepoch], verbose=1, workers=2)
        exit()
