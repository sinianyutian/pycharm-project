import numpy as np
from keras.models import Sequential
from keras.layers import Convolution2D, Activation, MaxPooling2D, Dense

from keras_module.hwh_layers import SpatialPyramidPooling

batch_size = 64
num_channels = 3
num_classes = 10

model = Sequential()

# uses theano ordering. Note that we leave the image size as None to allow multiple image sizes
model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(None, None,3)))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(64, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(SpatialPyramidPooling([1, 2, 4]))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='sgd')

# train on 64x64x3 images
# model.fit(np.random.rand(batch_size, 64, 64,num_channels), np.zeros((batch_size, num_classes)))
# train on 32x32x3 images
model.fit(np.random.rand(batch_size,  32, 32,num_channels), np.zeros((batch_size, num_classes)))