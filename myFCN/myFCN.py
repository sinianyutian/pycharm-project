import glob
import os
import pickle

from keras.callbacks import ModelCheckpoint

import pascal
import numpy as np
import matplotlib.pylab as plt

import my_utils
import fcn_model

weight_path = 'myFCN'
nb_epochs = 1000
batch_size = 32
data_type = 'trainval'
is_test = True

losses = []
acces = []

pasacl_train = pascal.PascalVOC2012SegmentationDataset(data_type)
pasacl_val = pascal.PascalVOC2012SegmentationDataset('val')
data_total_len = len(pasacl_train)
my_gen_train = my_utils.ImageIterator(pasacl_train, batch_size)
my_gen_val = my_utils.ImageIterator(pasacl_val, batch_size=100)

# load model
fcn32 = fcn_model.fcn32_model()
# fcn32 = fcn_model.fcn16_model()
# fcn32.summary()
# exit()


def test_img(pascal_iter, index):
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
    exit()
    return


# def load_data(file_name, index, batch_size):
#     # Loading data
#     pasacl_train = pascal.PascalVOC2012SegmentationDataset(file_name)
#     # test_img(pasacl_train,19)
#     datum, label = pasacl_train.get_example(19)
#     # print(np.max(label,axis=0))
#     # exit()
#
#     data_len = batch_size
#     train_shape = (data_len,) + datum.shape
#     label_shape = (data_len,) + label.shape
#     train_imgs = np.zeros(train_shape)
#     tmp_train_label = np.zeros(label_shape)
#     for i in range(data_len):
#         datum, label = pasacl_train.get_example(index * data_len + i)
#         train_imgs[i, :, :, :] = datum
#         tmp_train_label[i, :, :] = label
#     train_label_shape = label_shape + (21,)
#     train_label = np.zeros(train_label_shape)
#     for i in range(label_shape[0]):
#         for j in range(label_shape[1]):
#             for k in range(label_shape[2]):
#                 max_index = int(tmp_train_label[i, j, k])
#                 if max_index < 0:
#                     max_index = 0
#                     tmp_train_label[i, j, k] = 0
#                 train_label[i, j, k, max_index] = 1
#     return train_imgs, train_label


# print(train_imgs.shape)
# print(train_label.shape)
# exit()
# value = np.argmax(train_label, -1) - tmp_train_label
# print(np.max(value),np.min(value))
# print(train_shape)
# print(label_shape)
# print(train_label_shape)
# print(train_label.shape)


def draw_batch_images(generator, batch_size=5):
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


# for i in range(200):
#     print(i)
#     my_gen.next()
# for i in range(4):
#     draw_batch_images(my_gen)
# exit()

def draw_batch_label(datas, label_pred, label_true, batch_size=6):
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


def plot_loss(losses):
    # display.clear_output(wait=True)
    # display.display(plt.gcf())
    plt.figure(figsize=(10, 8))
    plt.plot(losses, label='fcn loss')
    plt.legend()
    plt.show()


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

# Train
try:
    newest = max(glob.iglob(weight_path + '*.hdf5'), key=os.path.getctime)
    fcn32.load_weights(newest)
    print("Load newest:  ")
    print("       " + newest)
finally:
    if is_test:
        # img_test, label_test = load_data(data_type, 12, batch_size=32)
        # fcn32.load_weights('myFCN-epoch.263-loss.0.1588.hdf5')
        for indexabc in range(5):
            img_test, label_test = my_gen_val.next()
            label_pred = fcn32.predict(img_test)
            draw_segment_pair(data_labels_pred=label_pred, data_labels=label_test)
            # draw_batch_label(img_test, label_pred=label_pred, label_true=label_test)
        exit()

    checkpointer = ModelCheckpoint(filepath=weight_path + '-epoch.{epoch:03d}-{val_loss:.4f}.hdf5',
                                   monitor='val_loss', verbose=1,  period=2)
    fcn32.fit_generator(generator=my_gen_train, steps_per_epoch=int(data_total_len / batch_size),
                        nb_epoch=1000, max_q_size=10, validation_data=my_gen_val,
                        validation_steps=3, callbacks=[checkpointer], verbose=1, workers=2)
    # print("load total samples:" + str(data_total_len))
    # nb_batches = int(data_total_len / batch_size)
    # progress_bar = Progbar(target=nb_batches)
    # val_image_batches, val_label_batches = load_data(data_type, 0, 3 * batch_size)
    # val_data_len = val_image_batches.shape[0]
    # print("load val samples:" + str(val_data_len))
    #
    # for epoch in range(nb_epochs):
    #     print('Epoch {} of {}'.format(epoch, nb_epochs))
    #     print("    ")
    #     for index in range(2, nb_batches):
    #         progress_bar.update(index)
    #
    #         # get a batch of real images
    #         image_batch, label_batch = load_data(data_type, index, batch_size)
    #
    #         loss = fcn32.train_on_batch(image_batch, label_batch)
    #         losses.append(loss)
    #
    #     acc = fcn32.test_on_batch(val_image_batches, val_label_batches)
    #     acces.append(acc[-1])
    #     print("    loss=" + str(losses[-1]))
    #     print("    acces=" + str(acces[-1]))
    #     # print("    ")
    #     # # Updates plots
    #     if epoch % plt_frq == plt_frq - 1:
    #         fcn32.save_weights(weight_path + "-epoch.{0}-loss.{1:.4f}.hdf5".format(epoch, losses[-1][0]))
    #
    # output = open('fcn_data.pkl', 'wb')
    # # Pickle dictionary using protocol 0.
    # pickle.dump(losses, output)
    # pickle.dump(acces, output)
    # output.close()
