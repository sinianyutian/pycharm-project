import os
from PIL import Image

os.environ['KERAS_BACKEND'] = 'tensorflow'
from keras.applications import vgg16
from keras.engine import Input
from keras.engine import Model
from keras.engine import merge
from keras.layers import MaxPooling2D, AveragePooling2D, Convolution2D, BatchNormalization, LeakyReLU
from keras.layers import UpSampling2D
from keras.optimizers import Adam
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
from keras import backend as K
from keras.utils.generic_utils import Progbar


def preprocess_image(image_path):
    img = load_img(image_path, target_size=(img_nrows, img_ncols))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg16.preprocess_input(img)
    return img


def deprocess_image(x):
    if K.image_dim_ordering() == 'th':
        x = x.reshape((3, img_nrows, img_ncols))
        x = x.transpose((1, 2, 0))
    else:
        x = x.reshape((img_nrows, img_ncols, 3))
    # Remove zero-center by mean pixel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x


style_path = "../images/fire.jpg"
content_path ="../images/didi.jpg"
img_nrows = 256
img_ncols = 256
batch_size = 16
nb_epoch = 2000
style_image = preprocess_image(style_path)
content_image= preprocess_image(content_path)

if K.image_dim_ordering() == 'th':
    image_shape = (3, img_nrows, img_ncols)
else:
    image_shape = (img_nrows, img_ncols, 3)


def build_model():
    x = Input(image_shape)
    img_1 = x
    img_2 = AveragePooling2D(pool_size=(2, 2))(img_1)
    img_3 = AveragePooling2D(pool_size=(4, 4))(img_1)
    img_4 = AveragePooling2D(pool_size=(8, 8))(img_1)
    img_5 = AveragePooling2D(pool_size=(16, 16))(img_1)

    def block(x, nb_channel):
        x = Convolution2D(nb_channel, 3, 3, border_mode='same', init='he_normal')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        x = Convolution2D(nb_channel, 3, 3, border_mode='same', init='he_normal')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        x = Convolution2D(nb_channel, 1, 1, border_mode='same', init='he_normal')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        return x

    def join(x_up, x_keep):
        x_up = UpSampling2D((2, 2))(x_up)
        x_up = BatchNormalization()(x_up)
        x_keep = BatchNormalization()(x_keep)
        x = merge([x_up, x_keep], mode='concat')
        return x

    img_1 = block(img_1, 8)
    img_2 = block(img_2, 8)
    img_3 = block(img_3, 8)
    img_4 = block(img_4, 8)
    img_5 = block(img_5, 8)

    img_4 = join(img_5, img_4)

    img_4 = block(img_4, 16)
    img_3 = join(img_4, img_3)

    img_3 = block(img_3, 24)
    img_2 = join(img_3, img_2)

    img_2 = block(img_2, 32)
    img_1 = join(img_2, img_1)

    img_1 = block(img_1, 40)
    y = block(img_1, 3)

    model = Model(x, y)

    return model


# model = build_model()
# model.fit()
# model.summary()
# exit()

def gram_matrix(x):
    assert K.ndim(x) == 3
    if K.image_dim_ordering() == 'th':
        features = K.batch_flatten(x)
    else:
        features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram


def style_loss(style, combination):
    assert K.ndim(style) == 3
    assert K.ndim(combination) == 3
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_nrows * img_ncols
    # return K.sum(K.square(S - C))
    return K.sum(K.square(S - C)) / (4. * (channels ** 2) * (size ** 2))


# an auxiliary loss function
# designed to maintain the "content" of the
# base image in the generated image
def content_loss(base, combination):
    return K.sum(K.square(combination - base))


# the 3rd loss function, total variation loss,
# designed to keep the generated image locally coherent
def total_variation_loss(x):
    assert K.ndim(x) == 4
    if K.image_dim_ordering() == 'th':
        a = K.square(x[:, :, :img_nrows - 1, :img_ncols - 1] - x[:, :, 1:, :img_ncols - 1])
        b = K.square(x[:, :, :img_nrows - 1, :img_ncols - 1] - x[:, :, :img_nrows - 1, 1:])
    else:
        a = K.square(x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, 1:, :img_ncols - 1, :])
        b = K.square(x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, :img_nrows - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))


def loss(y_true, y_pred):
    style_reference_image = K.variable(style_image)
    content_reference_image = K.variable(content_image)
    input_tensor = K.concatenate([y_pred,
                                  style_reference_image,content_reference_image], axis=0)
    new_model = vgg16.VGG16(input_tensor=input_tensor,
                            weights='imagenet', include_top=False)
    new_model.trainable = False
    outputs_dict = dict([(layer.name, layer.output) for layer in new_model.layers])
    layer_features = outputs_dict['block4_conv2']
    base_image_features = layer_features[2, :, :, :]
    combination_features = layer_features[0, :, :, :]
    out_loss = 0.025 * content_loss(base_image_features,
                                    combination_features)

    feature_layers = ['block1_conv1', 'block2_conv1',
                      'block3_conv1', 'block4_conv1',
                      'block5_conv1']

    # return out_loss
    # return K.mean(y_pred)
    # out_loss = K.variable(0.)
    for layer_name in feature_layers:
        layer_features = outputs_dict[layer_name]
        style_reference_features = layer_features[1, :, :, :]
        combination_features = layer_features[0, :, :, :]
        sl = style_loss(style_reference_features, combination_features)
        # return sl
        out_loss += sl
    out_loss += total_variation_loss(y_pred)
    return out_loss


model = build_model()
model.compile(Adam(0.1), loss=loss, metrics=['accuracy'])
model.summary()
progress_bar = Progbar(target=nb_epoch)
for index in range(nb_epoch):
    progress_bar.update(index)
    print("\n")
    if K.image_dim_ordering() == 'th':
        x = np.random.uniform(0, 1, (batch_size, 3, img_nrows, img_ncols))
    else:
        x = np.random.uniform(0, 1, (batch_size, img_nrows, img_ncols, 3))
        x += content_image
    model.fit(x, x, batch_size=batch_size, nb_epoch=1)
    if index % 100 == 1:
        x = np.random.uniform(0, 1, (1, img_nrows, img_ncols, 3))
        x += content_image
        model.save_weights("model-{0}.hdf5".format(index))
        y = model.predict(x)
        y = deprocess_image(y)
        Image.fromarray(y).save(
            'plot_{0}.png'.format(index))
