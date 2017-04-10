import numpy as np
from keras.applications import vgg16
from keras.preprocessing.image import load_img, img_to_array


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

