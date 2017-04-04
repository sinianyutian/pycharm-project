'''Deep Dreaming in Keras.

Run the script with:
```
python deep_dream.py path_to_your_base_image.jpg prefix_for_results
```
e.g.:
```
python deep_dream.py img/mypic.jpg results/dream
```

It is preferable to run this script on GPU, for speed.
If running on CPU, prefer the TensorFlow backend (much faster).

Example results: http://i.imgur.com/FX6ROg9.jpg
'''
from __future__ import print_function

import PIL
import tensorflow as tf
import os

import time

os.environ['KERAS_BACKEND'] = 'tensorflow'
from keras import backend as K

K.set_image_dim_ordering('tf')

from PIL import Image
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import argparse

from keras.applications import vgg16
from keras.layers import Input

import keras
keras.callbacks.TensorBoard

# parser = argparse.ArgumentParser(description='Style Transfer Dreams with Keras.')
# parser.add_argument('base_image_path', metavar='base', type=str,
#                     help='Path to the image to transform.')
# parser.add_argument('result_prefix', metavar='res_prefix', type=str,
#                     help='Prefix for the saved results.')
#
# args = parser.parse_args()
# base_image_path = args.base_image_path
# result_prefix = args.result_prefix
sess = tf.Session()
K.set_session(sess)
# Keras flag - we are not training, just testing now
K.set_learning_phase(0)

# dimensions of the generated picture.
img_height = 600
img_width = 600


def random_image():
    '''Generate a random noise image and return it as np array.
        Args:
            size: The desired dimensions of the output image.
            random_noise: Whether to generate a random noise
             image or to load a picture.
            filename: The fullpath to the image to load.
    '''
    # Generate a random noise image
    # 3 is the number of channels (RGB) in the image.
    dim_ordering = K.image_dim_ordering()
    if dim_ordering == 'tf':
        img = np.random.random(size=(img_height, img_width, 3))
    else:
        img = np.random.random(size=(3, img_height, img_width))

    img_array = img * 256
    return img_array


# util function to open, resize and format pictures into appropriate tensors
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(img_height, img_width))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg16.preprocess_input(img)
    return img

# util function to convert a tensor into a valid image
def deprocess_image(x):
    if K.image_dim_ordering() == 'th':
        x = x.reshape((3, img_height, img_width))
        x = x.transpose((1, 2, 0))
    else:
        x = x.reshape((img_height, img_width, 3))
    # Remove zero-center by mean pixel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def showImage(image,title="none"):
    real_image = deprocess_image(image)
    Image.fromarray(real_image).show(title=title)

# Image which will define the style
style_image = preprocess_image('images/fire.jpg')
# Image to define the content
content_image = preprocess_image('images/didi.jpg')

showImage(style_image,"style_image")
showImage(content_image,"content_image")

# path to the model weights file.
weights_path = 'vgg16_weights.h5'

if K.image_dim_ordering() == 'th':
    img_size = (3, img_height, img_width)
else:
    img_size = (img_height, img_width, 3)
# this will contain our generated image
dream = Input(batch_shape=(1,) + img_size)

image = random_image()
initial = np.expand_dims(image, axis=0).astype('float32')
input_tensor = tf.Variable(initial)
# input_tensor = K.variable(initial)
# build the VGG16 network with our placeholder
# the model will be loaded with pre-trained ImageNet weights
model = vgg16.VGG16(input_tensor=input_tensor,
                    weights='imagenet', include_top=False)
print('Model loaded.')
model.summary()

# Define the style layers
style_layers = [model.get_layer('block1_conv1').output,
                model.get_layer('block2_conv1').output,
                model.get_layer('block3_conv1').output,
                model.get_layer('block4_conv1').output,
                model.get_layer('block5_conv1').output]

# Define the content layers
content_layers = model.get_layer('block4_conv2').output

# Compute the style activations
style_layers_computed = sess.run(
    style_layers,
    feed_dict={input_tensor: style_image})

# Compute the content activations
content_layers_computed = sess.run(
    content_layers,
    feed_dict={input_tensor: content_image})


def style_loss(current, computed):
    '''Define the style loss between a tensor and an np array.
    Args:
        current: tf.Tensor. The style activations of the current image.
        computed: np array. The style activations of the style input image.
    '''
    style_losses = []
    for layer1, layer2 in zip(current, computed):
        _, height, width, number = map(lambda i: i, layer2.shape)
        size = height * width * number

        # Compute layer1 Gram matrix
        feats1 = tf.reshape(layer1, (-1, number))
        layer1_gram = tf.matmul(tf.transpose(feats1), feats1) / size
        # Compute layer2 Gram matrix
        feats2 = tf.reshape(layer2, (-1, number))
        layer2_gram = tf.matmul(tf.transpose(feats2), feats2) / size

        dim1, dim2 = map(lambda i: i.value, layer1_gram.get_shape())
        loss = tf.sqrt(tf.reduce_sum(tf.square((layer1_gram - layer2_gram) / (number * number))))
        style_losses.append(loss)
    return tf.add_n(style_losses)


def content_loss(current, computed):
    # Currently only for a single layer
    _, height, width, number = computed.shape
    size = height * width * number
    return tf.sqrt(tf.nn.l2_loss(current - computed) / size)


def total_variation_loss(image):
    dims = image.get_shape()
    tv_x_size = dims[1].value * (dims[2].value - 1) * dims[3].value
    tv_y_size = (dims[1].value - 1) * dims[2].value * dims[3].value

    return (
        tf.reduce_sum(tf.abs(image[:, 1:, :, :] - image[:, :-1, :, :])) +
        tf.reduce_sum(tf.abs(image[:, :, 1:, :] - image[:, :, :-1, :]))
    )


def setup_gradient(input_tensor, result_tensor):
    '''Setup the gradient of the input tensor w.t.r
    to the result tensor.
    Args:
        input_tensor: The input features tensor.
        result_tensor: The tensor that we want to maximize.
    '''
    # First get the result tensor mean
    excitement_score = tf.reduce_mean(result_tensor)

    # Gradients give us how to change the input (input_tensor)
    # to increase the excitement_score.
    # We get the first result only since the model is designed to
    # work on batches, and we only use single image.
    gradient = tf.gradients(excitement_score, input_tensor)[0]

    # Normalize the gradient by its L2 norm.
    # Disabled for now.
    # gradient /= (tf.sqrt(tf.reduce_mean(tf.square(gradient)))
    #                            + 1e-5)

    return gradient, excitement_score


# How much content, style and total variance loss contribute to the
# total loss.
content_weight = 1e3
style_weight = 1e6
tv_weight = 1e-3

# Set up the style, content, total variation, as well as total loss
# and use them to define the gradient.
with tf.variable_scope("style_loss") as scope:
    style_loss_op = style_weight * style_loss(style_layers, style_layers_computed)
with tf.variable_scope("content_loss") as scope:
    content_loss_op = content_weight * content_loss(content_layers, content_layers_computed)
with tf.variable_scope("tv_loss") as scope:
    tv_loss_op = tv_weight * total_variation_loss(input_tensor)
with tf.variable_scope("loss") as scope:
    total_loss_op = style_loss_op + content_loss_op + tv_loss_op
with tf.variable_scope("gradient") as scope:
    gradient_op, score_op = setup_gradient(input_tensor, total_loss_op)

# Start with a high learning rate initially.
# Adam will gradually decrease this.
learning_rate = 10
# Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate)
# Compute the gradients for a list of variables.
grads_and_vars = optimizer.compute_gradients(score_op, [input_tensor])
# Op that ask the optimizer to apply the gradients.
train_step = optimizer.apply_gradients(grads_and_vars)

with sess.as_default():
    variables = tf.global_variables()
    variables = list(variables)
    packed = [tf.is_variable_initialized(v) for v in variables]
    init_flag = sess.run(packed)
    uninitialized_vars= [v for v, f in zip(variables, init_flag) if not f]
    # Get uninitialized vars and their initializers
    initializers = [var.initializer for var in uninitialized_vars]

    # Print uninitialized variables
    print([initializer.name for initializer in initializers])

    # Initialize the variables
    _ = [initializer.run() for initializer in initializers]

with sess.as_default():
    # Define random 0 to 1 image with size (batch_size, image_size, channels)
    initial_random = tf.random_normal(mean=0.5, stddev=.5, shape=(1,) + (img_height, img_width) + (3,))
    # Use the content image
    initial_content = content_image
    # Init the input tensor
    input_tensor.initializer.run()
    # input_tensor.assign(tf.clip_by_value(initial_content * initial_random, 0, 255)).eval()
    input_tensor.assign(tf.clip_by_value(initial_random, 0, 255)).eval()
    # Recommended at least 500 - 1000 iterations for a good quality image.
    # Good style should be visible even after 100 iters.
    iterations = 1000
    # How many times to print the progress
    print_n_times = 10
    print_every_n = max(iterations // print_n_times, 1)

    # To compute total optimization time
    start_time = time.time()


    # Helper to print the losses.
    def print_progress(i,
                       loss_computed,
                       style_loss_computed,
                       content_loss_computed,
                       tv_loss_computed):
        print('Iteration %d/%d Content L: %g Style L: %g TV L: %g Total L: %g' % (
            i,
            iterations,
            content_loss_computed,
            style_loss_computed,
            tv_loss_computed,
            loss_computed
        ))


    # Keep only the image with the lowest loss
    # (in case we converge).
    best_loss = float('inf')
    best = None

    # Optimization loop
    for i in range(iterations):
        # Keep the input_tensor between 0 and 255
        # (gives slightly better output, slows optimization by factor of 2)
        # input_tensor.assign(tf.clip_by_value(input_tensor, 0, 255)).eval()

        # Run the training (train_step), and get the losses
        (_, result_image, loss_computed,
         style_loss_computed, content_loss_computed,
         tv_loss_computed) = sess.run(
            [train_step, input_tensor, score_op, style_loss_op, content_loss_op,
             tv_loss_op])

        # Print progress
        if i % print_every_n == 0:
            print_progress(i, loss_computed,
                           style_loss_computed, content_loss_computed,
                           tv_loss_computed)
            showImage(np.squeeze(result_image),title="{} iteration".format(i))

        if loss_computed < best_loss:
            best_loss = loss_computed
            best = result_image

    total_time = time.time() - start_time
    print('Training took {:.0f} seconds or {:.2f} s/iteration !'.format(
        total_time,
        total_time / iterations))

# best = np.clip(best, 0, 255)
print("last image")
showImage(np.squeeze(best),title="best")
