import os

import numpy as np
from keras import backend as K
from keras.preprocessing.image import Iterator, array_to_img
import copy


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

    def __init__(self, image_iter, batch_size=32,
                 is_list=False, shuffle=False, seed=None,
                 save_to_dir=None, save_prefix='', save_format='jpeg'):

        self.image_iter = image_iter
        tmp_data1, tmp_data2 = image_iter[0]
        # [x1,x2] [y1,y2,y3]
        self.is_list = is_list

        self.x_shape = np.asarray(tmp_data1).shape
        self.y_shape = np.asarray(tmp_data2).shape
        self.batch_size = batch_size

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
        # this line aims to ensure get batch has the same batch_size
        while len(index_array) != self.batch_size:
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
            batch_x[i] = np.asarray(x)
            batch_y[i] = np.asarray(y)
        # print(batch_x.shape)
        # print(batch_y.shape)
        # print(batch_x[:,1,:,:,:])
        # print(self.x_shape[0])
        if self.is_list:
            output_x = []
            output_y = []
            for index in range(self.x_shape[0]):
                output_x.append(batch_x[:, index, :, :, :])

            for index in range(self.y_shape[0]):
                output_y.append(batch_y[:, index, :, :, :])
            return output_x, output_y
        else:
            if self.save_to_dir:
                for i in range(current_batch_size):
                    img = array_to_img(batch_x[i], self.data_format, scale=True)
                    fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                      index=current_index + i,
                                                                      hash=np.random.randint(1e4),
                                                                      format=self.save_format)
                    img.save(os.path.join(self.save_to_dir, fname))
            return batch_x, batch_y
