from __future__ import absolute_import
from __future__ import print_function

import glob

import numpy as np
import os
import matplotlib.pylab as plt

from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.preprocessing.image import Iterator, array_to_img
from keras import backend as K

import fcn_model

try:
    from PIL import Image as pil_image
except ImportError:
    pil_image = None

facade_label_names = \
    np.array(['facade', 'molding', 'cornice', 'pillar',
              'window', 'door', 'sill', 'blind',
              'balcony', 'shop', 'deco', 'background'])


# download `BASE` dataset from http://cmp.felk.cvut.cz/~tylecr1/facade/
def load_facade(dataDir='./facade/base', img_size=224, data_range=(1, 300)):
    print("load dataset start")
    print("    from: %s" % dataDir)
    print("    range: [%d, %d)" % (data_range[0], data_range[1]))
    data_total_len = len(range(data_range[0], data_range[1]))
    data_img = np.zeros((data_total_len, img_size, img_size, 3))
    data_label = np.zeros((data_total_len, img_size, img_size, 12))
    data_index = 0
    for i in range(data_range[0], data_range[1]):
        img = pil_image.open(dataDir + "/cmp_b%04d.jpg" % i)
        label = pil_image.open(dataDir + "/cmp_b%04d.png" % i)
        img = img.resize((img_size, img_size), pil_image.BILINEAR)
        label = label.resize((img_size, img_size), pil_image.NEAREST)
        img = np.asarray(img).astype("f") / 128.0 - 1.0
        label_ = np.asarray(label) - 1  # [0, 12)
        label = np.zeros((12, img.shape[0], img.shape[1])).astype("i")
        for j in range(12):
            label[j, :] = label_ == j
        label = label.transpose((1, 2, 0))
        data_img[data_index, :, :, :] = img
        data_label[data_index, :, :, :] = label
        data_index += 1
    # print(np.max(data_img),np.min(data_img))
    # print(np.max(data_label),np.min(data_label))

    print("load dataset done")
    return data_img, data_label


def draw_facade_batch(data_imgs, data_labels, batch_size=5):
    plt.figure(figsize=(124, 124))
    for index in range(batch_size):
        datum = data_imgs[index]
        datum = (datum + 1.0) * 128.
        datum = datum.astype(np.uint8)

        label = data_labels[index]
        label = np.argmax(label, axis=-1)
        label += 1
        plt.subplot(2, batch_size, index + 1)
        plt.imshow(datum)
        plt.subplot(2, batch_size, batch_size + index + 1)
        plt.imshow(label)
    plt.show()
    return


def draw_facade_images(data_labels_pred, data_labels, batch_size=5):
    plt.figure(figsize=(124, 124))
    for index in range(batch_size):
        label_pred = data_labels_pred[index]
        label_pred = np.argmax(label_pred, axis=-1)
        label_pred += 1

        label = data_labels[index]
        label = np.argmax(label, axis=-1)
        label += 1
        plt.subplot(2, batch_size, index + 1)
        plt.imshow(label_pred)
        plt.subplot(2, batch_size, batch_size + index + 1)
        plt.imshow(label)
    plt.show()
    return


class ImageIterator(Iterator):
    """Iterator yielding data from a Numpy array.

    # Arguments
        imge_iter:image data which can get by index i and has len attr
            image,target = image_iter(i)
        shape:image shape(for example:(256,256,3))
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seed for data shuffling.
        save_to_dir: Optional directory where to save the pictures
            being yielded, in a viewable format. This is useful
            for visualizing the random transformations being
            applied, for debugging purposes.
        save_prefix: String prefix to use for saving sample
            images (if `save_to_dir` is set).
        save_format: Format to use for saving sample images
            (if `save_to_dir` is set).
    """

    def __init__(self, image_iter,
                 batch_size=32, shuffle=False, seed=None,
                 save_to_dir=None, save_prefix='', save_format='jpeg'):

        self.image_iter = image_iter
        tmp_data1, tmp_data2 = image_iter[0]
        self.x_shape = tmp_data1.shape
        self.y_shape = tmp_data2.shape

        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        self.data_total_len = len(image_iter)
        super(ImageIterator, self).__init__(self.data_total_len, batch_size, shuffle, seed)

    def next(self):
        """For python 2.x.

        # Returns
            The next batch.
        """
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch.
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        batch_x = np.zeros(tuple([current_batch_size] + list(self.x_shape)), dtype=K.floatx())
        batch_y = np.zeros(tuple([current_batch_size] + list(self.y_shape)), dtype=K.floatx())
        for i, j in enumerate(index_array):
            x, y = self.image_iter[j]
            # x = self.image_data_generator.random_transform(x.astype(K.floatx()))
            # x = self.image_data_generator.standardize(x)
            batch_x[i] = x
            batch_y[i] = y
        if self.save_to_dir:
            for i in range(current_batch_size):
                img = array_to_img(batch_x[i], self.data_format, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                  index=current_index + i,
                                                                  hash=np.random.randint(1e4),
                                                                  format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
        return batch_x, batch_y


if __name__ == '__main__':
    dir_path = '/home/andrew/facade/base'
    # draw_facade_batch(imgs, labels)
    # exit()
    weight_path = 'facade-'
    nb_epochs = 500
    batch_size = 32
    is_test = True
    facade = fcn_model.facade_model(224, 12)
    # facade.summary()
    # Train
    while True:
        try:
            newest = max(glob.iglob(weight_path + '*.hdf5'), key=os.path.getctime)
            facade.load_weights(newest)
            print("Load newest:  ")
            print("       " + newest)
        finally:
            if is_test:
                imgs_val, labels_val = load_facade(dataDir=dir_path, data_range=(20, 29))
                labels_pred = facade.predict(imgs_val)
                draw_facade_images(labels_pred, labels_val)
                exit()
            imgs, labels = load_facade(dataDir=dir_path, data_range=(1, 320))
            imgs_val, labels_val = load_facade(dataDir=dir_path, data_range=(320, 378))

            checkpointer = ModelCheckpoint(filepath=weight_path + '.{epoch:03d}-{val_loss:.4f}.hdf5',
                                           monitor='val_loss', verbose=1, period=100)
            tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0)
            facade.fit(imgs, labels, epochs=50000, validation_data=(imgs_val, labels_val),
                       batch_size=batch_size, shuffle=True, callbacks=[checkpointer, tensorboard])
