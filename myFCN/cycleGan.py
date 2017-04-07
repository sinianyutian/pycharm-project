import glob, os

import time

import skimage
from keras.engine import Input
from keras.engine import Model
from keras.engine.training import GeneratorEnqueuer
from keras.losses import mean_absolute_error
from keras.metrics import binary_accuracy
from keras.optimizers import Adam
import keras.backend as K
from keras.utils import Progbar

from fcn_model import pix2pix_discriminator_single, pix2pix_generator_single
import fcn_utils
from my_utils import ImageIterator
from pascal_unpair import PascalVOC2012SegmentationDatasetUnpair
import numpy as np
import matplotlib.pylab as plt


def cycle_loss(y_true, y_pred):
    return K.categorical_crossentropy(y_pred, y_true, from_logits=True)


def cycle_acc(y_true, y_pred):
    return K.mean(K.cast(K.equal(K.argmax(y_true, axis=-1),
                                 K.argmax(y_pred, axis=-1)),
                         K.floatx()))


def build_model(input_size=(256, 256)):
    netG = pix2pix_generator_single(nb_in=3, nb_out=21, input_size=input_size)
    netF = pix2pix_generator_single(nb_in=21, nb_out=3, input_size=input_size)

    netDy = pix2pix_discriminator_single(nb_in=21, input_size=input_size)
    netDx = pix2pix_discriminator_single(nb_in=3, input_size=input_size)
    adamDy = Adam()
    adamDx = Adam()
    netDy.compile(optimizer=adamDy, loss=mean_absolute_error, metrics=[binary_accuracy])
    netDx.compile(optimizer=adamDx, loss=mean_absolute_error, metrics=[binary_accuracy])

    netDy.trainable = False
    netDx.trainable = False

    x = Input(input_size + (3,))
    y = Input(input_size + (21,))

    y_ = netG(x)
    x_out = netF(y_)
    y_true = netDy(y_)

    x_ = netF(y)
    y_out = netG(x_)
    x_true = netDx(x_)

    netCycle = Model([x, y], [x_out, y_true, y_out, x_true])
    adamCycle = Adam()
    #notice: model_2->cycle_acc has no meaning!
    netCycle.compile(optimizer=adamCycle,
                     loss=[mean_absolute_error, mean_absolute_error, cycle_loss, mean_absolute_error],
                     metrics={'model_2': cycle_acc, 'model_3': binary_accuracy,
                              'model_1': cycle_acc, 'model_4': binary_accuracy}
                     )
    return netG, netF, netDy, netDx, netCycle


def draw_batch_images(my_gen_datas, my_gen_labels, batch_size=5):
    plt.figure(figsize=(124, 124))
    for index in range(batch_size):
        datum = my_gen_datas[index]
        datum *= 255.
        label = my_gen_labels[index]
        label = np.argmax(label, axis=-1)
        plt.subplot(2, batch_size, index + 1)
        plt.imshow(datum.astype(np.uint8))
        plt.subplot(2, batch_size, batch_size + index + 1)
        plt.imshow(label)
    plt.show()
    return


# test model
# netG, netF, netDy, netDx, netCycle = build_model(input_size=(128, 128))
# netCycle.summary()
# exit()

if __name__ == '__main__':
    weight_path = 'cycleGAN-'
    nb_epochs = 1000
    batch_size = 32
    data_type = 'trainval'
    is_test = False
    plt_frq = 20
    # size % 32==0
    image_size = (192, 192)

    max_q_size = 10
    workers = 2
    pickle_safe = False
    wait_time = 0.1

    losses = []
    acces = []

    pasacl_train = PascalVOC2012SegmentationDatasetUnpair(data_type, target_size=image_size)
    data_total_len = len(pasacl_train)
    my_gen_train = ImageIterator(pasacl_train, batch_size)
    # for _ in range(5):
    #     tmp_x, tmp_y = next(my_gen_train)
    #     draw_batch_images(tmp_x, tmp_y)
    # exit()

    # build model
    netG, netF, netDy, netDx, netCycle = build_model(input_size=image_size)
    patch_size = netDx.output_shape[1]

    # Train
    try:
        newest = max(glob.iglob(weight_path + 'netG*.hdf5'), key=os.path.getctime)
        netG.load_weights(newest)
        print("Load newest netG:  ")
        print("       " + newest)

        newest = max(glob.iglob(weight_path + 'netF*.hdf5'), key=os.path.getctime)
        netF.load_weights(newest)
        print("Load newest netF:  ")
        print("       " + newest)

        newest = max(glob.iglob(weight_path + 'netDy*.hdf5'), key=os.path.getctime)
        netDy.load_weights(newest)
        print("Load newest netDy:  ")
        print("       " + newest)

        newest = max(glob.iglob(weight_path + 'netDx*.hdf5'), key=os.path.getctime)
        netDx.load_weights(newest)
        print("Load newest netDx:  ")
        print("       " + newest)

        newest = max(glob.iglob(weight_path + 'netCycle*.hdf5'), key=os.path.getctime)
        netCycle.load_weights(newest)
        print("Load newest netCycle:  ")
        print("       " + newest)

    finally:
        if is_test:
            for i in range(5):
                x_test, y_test = my_gen_train.next()
                y_pred = netG.predict(x_test)
                draw_batch_images(x_test, y_pred)

                x_pred = netF.predict(x_test)
                draw_batch_images(x_pred, y_test)
            exit()

        print("load total samples:" + str(data_total_len))
        nb_batches = int(data_total_len / batch_size)
        progress_bar = Progbar(target=nb_batches)

        try:
            enqueuer = GeneratorEnqueuer(my_gen_train, pickle_safe=pickle_safe)
            enqueuer.start(max_q_size=max_q_size, workers=workers)

            true_dis_sample = np.ones((batch_size, patch_size, patch_size, 1))
            false_dis_sample = np.zeros((batch_size, patch_size, patch_size, 1))
            true_false_dis_sample = np.concatenate([true_dis_sample, false_dis_sample])

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
                    x_batch, y_batch = generator_output

                    batch_size_geted = x_batch.shape[0]
                    if batch_size_geted != batch_size:
                        true_dis = np.ones((batch_size_geted, patch_size, patch_size, 1))
                        false_dis = np.zeros((batch_size_geted, patch_size, patch_size, 1))
                        true_false_dis = np.concatenate([true_dis, false_dis])
                    else:
                        true_dis = true_dis_sample
                        false_dis = false_dis_sample
                        true_false_dis = true_false_dis_sample

                    # train netG and netD
                    y_pred_batch = netG.predict(x_batch)
                    y_batches = np.concatenate([y_batch, y_pred_batch])
                    loss_Dy = netDy.train_on_batch(y_batches, true_false_dis)

                    x_pred_batch = netF.predict(y_batch)
                    x_batches = np.concatenate([x_batch, x_pred_batch])
                    loss_Dx = netDx.train_on_batch(x_batches, true_false_dis)
                    # print(x_batch.shape)
                    # print(y_batch.shape)
                    # print(true_dis.shape)
                    # print('train netCycle')

                    # # train netCycle
                    loss_Cycle = netCycle.train_on_batch([x_batch, y_batch],
                                                         [x_batch, true_dis, y_batch, true_dis])
                    # print("x_batch")
                    # print(np.max(x_batch), np.min(x_batch))
                    # x_pred = netG.predict(x_batch)
                    # print("x_pred")
                    # print(np.max(x_pred), np.min(x_pred))
                    # need to test loss_cycle shape
                    # print(loss_Cycle)
                    #
                    # print(x_batches.shape)
                    # print(y_batches.shape)
                    # print(true_false_dis.shape)
                    # exit()

                print("   ")
                print('{0:<22s} | {1:6s} | {2:<10s} | {3:6s} | {4:<15s}'.format(
                    'component', 'D_loss', 'D_accuracy', 'G_loss', 'G_accuracy'))
                print('-' * 65)

                ROW_FMT = '{0:<22s} | {1:<6.2f} | {2:<10.2f} | {3:<6.2f} | {4:<15.2f}'
                # need to get  right sequence
                print(ROW_FMT.format('Dy and x->GF->xp',
                                     loss_Cycle[2], loss_Cycle[6], loss_Cycle[1], loss_Cycle[5]))
                print(ROW_FMT.format('Dy', loss_Dy[0], loss_Dy[1], 0.0, 0.0))
                print(ROW_FMT.format('Dx and y->FG->yp)',
                                     loss_Cycle[4], loss_Cycle[8], loss_Cycle[3], loss_Cycle[7]))
                print(ROW_FMT.format('Dx', loss_Dx[0], loss_Dx[1], 0.0, 0.0))

                # Updates plots
                if epoch % plt_frq == plt_frq - 1:
                    netG.save_weights(
                        weight_path + "netG-epoch.{0}-loss.{1:.4f}.hdf5".format(epoch, loss_Cycle[1]))
                    netF.save_weights(
                        weight_path + "netF-epoch.{0}-loss.{1:.4f}.hdf5".format(epoch, loss_Cycle[3]))
                    netDy.save_weights(weight_path +
                                       "netDy-epoch.{0}-loss.{1:.4f}.hdf5".format(epoch, loss_Dy[0]))
                    netDx.save_weights(weight_path +
                                       "netDx-epoch.{0}-loss.{1:.4f}.hdf5".format(epoch, loss_Dx[0]))
                    netCycle.save_weights(weight_path +
                                          "netCycle-epoch.{0}-loss.{1:.4f}.hdf5".format(epoch, loss_Cycle[0]))
        finally:
            if enqueuer is not None:
                enqueuer.stop()
