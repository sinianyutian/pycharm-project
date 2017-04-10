import matplotlib.pylab as plt

import numpy as np

from ..image_utils import deprocess_image


def test_img(pascal_iter, index, pasacl_train):
    datum, label = pascal_iter.get_example(index)
    img = pasacl_train.visualize_example(index)
    print(img.shape)
    plt.figure()
    plt.imshow(pasacl_train.datum_to_img(datum))
    plt.show()
    plt.figure()
    plt.imshow(img)
    plt.show()

    print(datum.shape)
    print(label.shape)
    return


def draw_batch_images(generator, pasacl_train, batch_size=5):
    my_gen_datas, my_gen_labels = generator.next()
    plt.figure(figsize=(124, 124))
    for index in range(batch_size):
        datum = my_gen_datas[index]
        label = my_gen_labels[index]
        label = np.argmax(label, axis=-1)
        img_pred = pasacl_train.visualize_pairs(datum, label)
        plt.subplot(2, batch_size, index + 1)
        plt.imshow(pasacl_train.datum_to_img(datum))
        plt.subplot(2, batch_size, batch_size + index + 1)
        plt.imshow(img_pred)
    plt.show()
    return


def draw_images_pair(img1_datas, img2_datas, index_pro=1, batch_size=5, is_save=True, prefix='st-',is_block=False):
    plt.figure(figsize=(100, 40))
    for index in range(batch_size):
        datum = img1_datas[index].copy()
        datum = deprocess_image(datum)
        label = img2_datas[index].copy()
        label = deprocess_image(label)
        plt.subplot(2, batch_size, index + 1)
        plt.imshow(datum)
        plt.subplot(2, batch_size, batch_size + index + 1)
        plt.imshow(label)
    if is_save:
        plt.savefig(prefix + str(index_pro) + '.jpg')
    else:
        plt.show(block=is_block)
    return


def draw_batch_label(datas, label_pred, label_true, pasacl_train, batch_size=6):
    plt.figure(figsize=(124, 124))
    for inner_index in range(batch_size):
        datum = datas[inner_index]
        label_pred_datum = label_pred[inner_index]
        label_pred_datum = np.argmax(label_pred_datum, axis=-1)
        label_true_datum = label_true[inner_index]
        label_true_datum = np.argmax(label_true_datum, axis=-1)
        tmp_img_pred = pasacl_train.visualize_pairs(datum, label_pred_datum)
        tmp_img_true = pasacl_train.visualize_pairs(datum, label_true_datum)
        plt.subplot(2, batch_size, inner_index + 1)
        plt.imshow(tmp_img_true)
        plt.subplot(2, batch_size, batch_size + inner_index + 1)
        plt.imshow(tmp_img_pred)
    plt.show()
    return


def draw_segment_pair(data_labels_pred, data_labels, batch_size=5):
    plt.figure(figsize=(124, 124))
    for index in range(batch_size):
        label_pred = data_labels_pred[index]
        label_pred = np.argmax(label_pred, axis=-1)
        # label_pred += 1

        label = data_labels[index]
        label = np.argmax(label, axis=-1)
        # label += 1
        plt.subplot(2, batch_size, index + 1)
        plt.imshow(label_pred)
        plt.subplot(2, batch_size, batch_size + index + 1)
        plt.imshow(label)
    plt.show()
    return
