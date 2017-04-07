import glob
import os
import pickle

import time
from keras.engine import Input
from keras.engine import Model
from keras.engine.training import GeneratorEnqueuer
from keras.losses import mean_absolute_error
from keras.metrics import binary_accuracy
from keras.optimizers import Adam
from keras.utils import Progbar
import numpy as np

import fcn_model
import pascal
import my_utils
import matplotlib.pylab as plt


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


# print('{0:<12s} | {1:4s} | {2:5s} | {3:5s} | {4:5s}'.format(
#     'component', 'D_loss', 'D_accuracy', 'G_loss', 'G_accuracy'))
# print('-' * 65)
# exit()

y = Input((256, 256, 21))
y_cdt = Input((256, 256, 3))
pixD = fcn_model.pix2pix_discriminator(nb_in=21, nb_out=3)

pixD_out = pixD([y, y_cdt])
pixD_model = Model([y, y_cdt], pixD_out)
adamD = Adam()
pixD_model.compile(optimizer=adamD, loss=mean_absolute_error, metrics=[binary_accuracy])
# pixD_model.summary()

pixD.trainable = False
pixG = fcn_model.pix2pix_generator(3, 21)
x = Input((256, 256, 3))
y = pixG(x)
pixG_model = Model(x, y)
adamG = Adam()
pixG_model.compile(optimizer=adamG, loss=fcn_model.fcn32_loss, metrics=[fcn_model.fcn32_acc])

y_cdt = Input((256, 256, 3))
pixD_out = pixD([y, y_cdt])

pix_cbd_model = Model([x, y_cdt], [pixD_out, y])
adam_cbd = Adam()
pix_cbd_model.compile(optimizer=adam_cbd, loss=[mean_absolute_error, fcn_model.fcn32_loss],
                      metrics=[binary_accuracy, fcn_model.fcn32_acc])


# pix_cbd_model.summary()


def load_data(file_name, index, batch_size):
    # Loading data
    pasacl_train_tmp = pascal.PascalVOC2012SegmentationDataset(file_name)
    # test_img(pasacl_train,19)
    datum, label = pasacl_train_tmp.get_example(19, target_size=(256, 256))
    # print(np.max(label,axis=0))
    # exit()

    data_len = batch_size
    train_shape = (data_len,) + datum.shape
    label_shape = (data_len,) + label.shape
    train_imgs = np.zeros(train_shape)
    tmp_train_label = np.zeros(label_shape)
    for i in range(data_len):
        datum, label = pasacl_train_tmp.get_example(index * data_len + i, target_size=(256, 256))
        train_imgs[i, :, :, :] = datum
        tmp_train_label[i, :, :] = label
    train_label_shape = label_shape + (21,)
    train_label = np.zeros(train_label_shape)
    for i in range(label_shape[0]):
        for j in range(label_shape[1]):
            for k in range(label_shape[2]):
                max_index = int(tmp_train_label[i, j, k])
                if max_index < 0:
                    max_index = 0
                    tmp_train_label[i, j, k] = 0
                train_label[i, j, k, max_index] = 1
    return train_imgs, train_label


weight_path = 'pix2pix'
nb_epochs = 500
batch_size = 32
data_type = 'trainval'
is_test = True
plt_frq = 10

max_q_size = 10
workers = 2
pickle_safe = False
wait_time = 0.1

train_val = []

pasacl_train = pascal.PascalVOC2012SegmentationDataset(data_type, target_size=(256, 256))
pasacl_val = pascal.PascalVOC2012SegmentationDataset('val', target_size=(256, 256))

my_gen_train = my_utils.ImageIterator(pasacl_train, batch_size)
my_gen_val = my_utils.ImageIterator(pasacl_val, batch_size=batch_size)

data_total_len = len(pasacl_train)
data_val_len = len(pasacl_val)

# Train
try:
    newest = max(glob.iglob(weight_path + 'pixG*.hdf5'), key=os.path.getctime)
    pixG_model.load_weights(newest)
    print("Load newest pixG:  ")
    print("       " + newest)

    newest = max(glob.iglob(weight_path + 'pixD*.hdf5'), key=os.path.getctime)
    pixD_model.load_weights(newest)
    print("Load newest pixD:  ")
    print("       " + newest)

    newest = max(glob.iglob(weight_path + 'pix-cbd*.hdf5'), key=os.path.getctime)
    pix_cbd_model.load_weights(newest)
    print("Load newest pix_cbd:  ")
    print("       " + newest)
finally:
    if is_test:
        for i in range(5):
            img_test, label_test = my_gen_val.next()
            label_pred = pixG_model.predict(img_test)
            draw_batch_label(img_test, label_pred=label_pred, label_true=label_test)
        exit()
    print("load total samples:" + str(data_total_len))
    print("load total val samples:" + str(data_val_len))
    nb_batches = int(data_total_len / batch_size)
    progress_bar = Progbar(target=nb_batches)
    # val_image_batches, val_label_batches = load_data('val', 0, 50)
    # val_data_len = val_image_batches.shape[0]
    # print("load val samples:" + str(val_data_len))
    try:
        enqueuer = GeneratorEnqueuer(my_gen_train, pickle_safe=pickle_safe)
        enqueuer.start(max_q_size=max_q_size, workers=workers)

        enqueuer_val = GeneratorEnqueuer(my_gen_val, pickle_safe=pickle_safe)
        enqueuer_val.start(max_q_size=max_q_size, workers=workers)

        true_dis_sample = np.ones((batch_size, 30, 30, 1))
        false_dis_sample = np.zeros((batch_size, 30, 30, 1))
        true_false_dis_sample = np.concatenate([true_dis_sample, false_dis_sample])
        # train discriminator
        # for index in range(100):
        #     generator_output = None
        #     while enqueuer.is_running():
        #         if not enqueuer.queue.empty():
        #             generator_output = enqueuer.queue.get()
        #             break
        #         else:
        #             time.sleep(wait_time)
        #     image_batch, label_batch = generator_output
        #     batch_size_geted = image_batch.shape[0]
        #     if batch_size_geted!=batch_size:
        #         true_dis = np.ones((batch_size_geted, 30, 30, 1))
        #         false_dis = np.zeros((batch_size_geted, 30, 30, 1))
        #         true_false_dis = np.concatenate([true_dis, false_dis])
        #     else :
        #         true_dis = true_dis_sample
        #         false_dis = false_dis_sample
        #         true_false_dis = true_false_dis_sample
        #
        #     # image_batch, label_batch = load_data(data_type, index, batch_size)
        #     label_pred_batch = pixG_model.predict(image_batch)
        #     image_batches = np.concatenate([image_batch, image_batch])
        #     label_batches = np.concatenate([label_batch, label_pred_batch])
        #     print(image_batches.shape)
        #     print(label_batches.shape)
        #     print(true_false_dis.shape)
        #     pixD_value = pixD_model.train_on_batch([label_batches, image_batches], true_false_dis)
        #     print(index,"  ",pixD_value[0], pixD_value[1])
        #
        # print("OK")
        # exit()
        for epoch in range(nb_epochs):
            print('Epoch {} of {}'.format(epoch, nb_epochs))
            print("    ")

            for index in range(0, nb_batches):
                progress_bar.update(index)
                generator_output = None
                while enqueuer.is_running():
                    if not enqueuer.queue.empty():
                        generator_output = enqueuer.queue.get()
                        break
                    else:
                        time.sleep(wait_time)
                image_batch, label_batch = generator_output

                batch_size_geted = image_batch.shape[0]
                if batch_size_geted != batch_size:
                    true_dis = np.ones((batch_size_geted, 30, 30, 1))
                    false_dis = np.zeros((batch_size_geted, 30, 30, 1))
                    true_false_dis = np.concatenate([true_dis, false_dis])
                else:
                    true_dis = true_dis_sample
                    false_dis = false_dis_sample
                    true_false_dis = true_false_dis_sample

                # image_batch, label_batch = load_data(data_type, index, batch_size)
                label_pred_batch = pixG_model.predict(image_batch)
                image_batches = np.concatenate([image_batch, image_batch])
                label_batches = np.concatenate([label_batch, label_pred_batch])

                # print(image_batches.shape)
                # print(label_batches.shape)
                # print(true_false_dis.shape)
                # exit()

                # train_discriminator
                pixD_value = pixD_model.train_on_batch([label_batches, image_batches], true_false_dis)

                # train_generator
                pix_cbd_value = pix_cbd_model.train_on_batch([image_batch, image_batch], [true_dis, label_batch])

            while enqueuer_val.is_running():
                if not enqueuer_val.queue.empty():
                    generator_val_output = enqueuer_val.queue.get()
                    break
                else:
                    time.sleep(wait_time)

            # validation
            val_image_batches, val_label_batches = generator_val_output

            batch_size_geted = val_image_batches.shape[0]
            if batch_size_geted != batch_size:
                true_dis = np.ones((batch_size_geted, 30, 30, 1))
                false_dis = np.zeros((batch_size_geted, 30, 30, 1))
                true_false_dis = np.concatenate([true_dis, false_dis])
            else:
                true_dis = true_dis_sample
                false_dis = false_dis_sample
                true_false_dis = true_false_dis_sample
            # print(val_image_batches.shape)
            # print(val_label_batches.shape)
            test_value = pix_cbd_model.test_on_batch([val_image_batches, val_image_batches],
                                                     [true_dis, val_label_batches])

            label_val_pred_batch = pixG_model.predict(val_image_batches)
            image_val_batches = np.concatenate([val_image_batches, val_image_batches])
            label_val_batches = np.concatenate([val_label_batches, label_val_pred_batch])
            pixD_test_value = pixD_model.test_on_batch([label_val_batches, image_val_batches], true_false_dis)

            print("   ")
            print('{0:<22s} | {1:6s} | {2:<10s} | {3:6s} | {4:<15s}'.format(
                'component', 'D_loss', 'D_accuracy', 'G_loss', 'G_accuracy'))
            print('-' * 65)

            ROW_FMT = '{0:<22s} | {1:<6.2f} | {2:<10.2f} | {3:<6.2f} | {4:<15.2f}'
            print(ROW_FMT.format('generator (train)',
                                 pix_cbd_value[1], pix_cbd_value[3], pix_cbd_value[2], pix_cbd_value[6]))
            print(ROW_FMT.format('generator (test)',
                                 test_value[1], test_value[3], test_value[2], test_value[6]))

            print(ROW_FMT.format('discriminator (train)',
                                 pixD_value[0], pixD_value[1], 0.0, 0.0))
            print(ROW_FMT.format('discriminator (test)',
                                 pixD_test_value[0], pixD_test_value[1], 0.0, 0.0))

            # print("    ")
            # # Updates plots
            if epoch % plt_frq == plt_frq - 1:
                pixG_model.save_weights(weight_path + "pixG-epoch.{0}-loss.{1:.4f}.hdf5".format(epoch, test_value[0]))
                pixD_model.save_weights(weight_path + "pixD-epoch.{0}-loss.{1:.4f}.hdf5".format(epoch, pixD_value[0]))
                pix_cbd_model.save_weights(weight_path +
                                           "pix-cbd-epoch.{0}-loss.{1:.4f}.hdf5".format(epoch, pix_cbd_value[0]))
            output = open('fcn_data.pkl', 'wb')
            # Pickle dictionary using protocol 0.
            pickle.dump(train_val, output)
            output.close()
    finally:
        if enqueuer is not None:
            enqueuer.stop()
        if enqueuer_val is not None:
            enqueuer_val.stop()
