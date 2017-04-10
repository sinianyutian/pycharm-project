from keras.applications import vgg16
from keras.engine import Input
from keras.engine import Model
from keras.layers import Conv2D, Deconv2D, Lambda

from keras_module.hwh_layers import AdaptiveInstanceNormalization


def build_vggConv(input_shape=(224, 224, 3),my_vgg=None):
    x1 = Input(input_shape)
    x2 = Input(input_shape)
    if my_vgg:
        vgg_model = my_vgg
    else:
        vgg_model = vgg16.VGG16(include_top=False, weights='imagenet')
        vgg_model.trainable = False

    for layer_tmp in vgg_model.layers:
        layer_tmp.trainable = False

    y1 = vgg_model(x1)
    y2 = vgg_model(x2)

    # Ada-IN
    y = AdaptiveInstanceNormalization()([y1, y2])

    # vgg deconv
    # Block 6
    y = Deconv2D(512, (3, 3), strides=(2, 2), activation='relu', padding='same')(y)
    y = Conv2D(512, (3, 3), activation='relu', padding='same', name='block6_conv1')(y)
    y = Conv2D(512, (3, 3), activation='relu', padding='same', name='block6_conv2')(y)

    # Block 7
    y = Deconv2D(512, (3, 3), strides=(2, 2), activation='relu', padding='same')(y)
    y = Conv2D(512, (3, 3), activation='relu', padding='same', name='block7_conv1')(y)
    y = Conv2D(512, (3, 3), activation='relu', padding='same', name='block7_conv2')(y)

    # Block 8
    y = Deconv2D(256, (3, 3), strides=(2, 2), activation='relu', padding='same')(y)
    y = Conv2D(256, (3, 3), activation='relu', padding='same', name='block8_conv1')(y)
    y = Conv2D(256, (3, 3), activation='relu', padding='same', name='block8_conv2')(y)

    # Block 9
    y = Deconv2D(128, (3, 3), strides=(2, 2), activation='relu', padding='same')(y)
    y = Conv2D(128, (3, 3), activation='relu', padding='same', name='block9_conv1')(y)

    # Block 10
    y = Deconv2D(64, (3, 3), strides=(2, 2), activation='relu', padding='same')(y)
    y = Conv2D(64, (3, 3), activation='relu', padding='same', name='block10_conv1')(y)
    y = Conv2D(3, (3, 3), activation='tanh', padding='same', name='block10_conv2')(y)

    y1 = Lambda(lambda x: x * 128.)(y)
    y2 = Lambda(lambda x: x * 128.)(y)
    y3 = Lambda(lambda x: x * 128.)(y)

    return Model([x1, x2], [y1, y2, y3])


if __name__ == '__main__':
    model = build_vggConv()
    model.summary()
