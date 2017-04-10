import matplotlib.pylab as plt

import numpy as np


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
