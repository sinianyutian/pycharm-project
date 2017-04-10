import os.path as osp
import os

import numpy as np
import scipy.misc

from keras.applications import vgg16
from keras.preprocessing import image
from keras.preprocessing.image import load_img

from segmentation_dataset import SegmentationDatasetBase


class PascalVOC2012SegmentationDataset(SegmentationDatasetBase):
    label_names = np.array([
        'background',
        'aeroplane',
        'bicycle',
        'bird',
        'boat',
        'bottle',
        'bus',
        'car',
        'cat',
        'chair',
        'cow',
        'diningtable',
        'dog',
        'horse',
        'motorbike',
        'person',
        'potted plant',
        'sheep',
        'sofa',
        'train',
        'tv/monitor',
    ])
    mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])

    def __init__(self, data_type, target_size=(224, 224)):
        # get ids for the data_type
        dataset_dir = '/home/andrew/VOC2012dev/VOCdevkit/VOC2012'
        imgsets_file = osp.join(
            dataset_dir,
            'ImageSets/Segmentation/{}.txt'.format(data_type))
        self.files = []
        for data_id in open(imgsets_file).readlines():
            data_id = data_id.strip()
            img_file = osp.join(
                dataset_dir, 'JPEGImages/{}.jpg'.format(data_id))
            label_rgb_file = osp.join(
                dataset_dir, 'SegmentationClass/{}.png'.format(data_id))
            self.files.append({
                'img': img_file,
                'label_rgb': label_rgb_file,
            })
        self.target_size = target_size

    def __len__(self):
        return len(self.files)

    def get_example(self, i):
        data_file = self.files[i]
        # load image
        img_file = data_file['img']
        # img = scipy.misc.imread(img_file, mode='RGB')
        img = load_img(img_file, target_size=self.target_size)
        img = image.img_to_array(img)
        datum = self.img_to_datum(img)
        # load label
        label_rgb_file = data_file['label_rgb']
        label_rgb = load_img(label_rgb_file, target_size=self.target_size)
        label_rgb = image.img_to_array(label_rgb)
        label_rgb = label_rgb.astype(np.uint8)
        # label_rgb = scipy.misc.imread(label_rgb_file, mode='RGB')
        label = self.label_rgb_to_32sc1(label_rgb)
        return datum, label

    def __getitem__(self, key):
        datum, label = self.get_example(key)
        label_shape = label.shape + (21,)
        label_data = np.zeros(label_shape)
        for i in range(label_shape[0]):
            for j in range(label_shape[1]):
                max_index = int(label[i, j])
                if max_index < 0:
                    max_index = 0
                label_data[i, j, max_index] = 1
        return datum, label_data


class PascalVOC2012SegmentationDatasetImagePair(PascalVOC2012SegmentationDataset):
    def __init__(self, data_type, target_size=(224, 224)):
        super(PascalVOC2012SegmentationDatasetImagePair, self).__init__(data_type, target_size)

    def get_example(self, i):
        data_file = self.files[i]
        # load image
        img_file = data_file['img']
        # img = scipy.misc.imread(img_file, mode='RGB')
        img = load_img(img_file, target_size=self.target_size)
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)

        img = vgg16.preprocess_input(img)
        datum = np.squeeze(img)
        # datum = img / 255.
        return datum, datum

    def __getitem__(self, key):
        return self.get_example(key)


class PascalVOC2012SegmentationDatasetUnpair(PascalVOC2012SegmentationDataset):
    def __init__(self, data_type, target_size=(224, 224)):
        super(PascalVOC2012SegmentationDatasetUnpair, self).__init__(data_type, target_size)

    def get_example(self, i):
        shuffle_int = np.random.randint(0, len(self.files), 1)
        data_file = self.files[i]
        # load image
        img_file = data_file['img']
        # img = scipy.misc.imread(img_file, mode='RGB')
        img = load_img(img_file, target_size=self.target_size)
        img = image.img_to_array(img)
        datum = img / 255.
        # datum = self.img_to_datum(img)
        # load label
        data_file = self.files[shuffle_int]
        label_rgb_file = data_file['label_rgb']
        label_rgb = load_img(label_rgb_file, target_size=self.target_size)
        label_rgb = image.img_to_array(label_rgb)
        label_rgb = label_rgb.astype(np.uint8)
        # label_rgb = scipy.misc.imread(label_rgb_file, mode='RGB')
        label = self.label_rgb_to_32sc1(label_rgb)
        return datum, label


class Content_Style_Pair(PascalVOC2012SegmentationDataset):
    def __init__(self, style_dir, data_type, target_size=(224, 224)):
        super(Content_Style_Pair, self).__init__(data_type, target_size)
        self.style_files = []
        lists = os.listdir(style_dir)
        for list_tmp in lists:
            self.style_files.append(style_dir + '/' + list_tmp)

    def style_len(self):
        return len(self.style_files)

    def get_example(self, i):
        data_file = self.files[i]
        # load image
        img_file = data_file['img']
        # img = scipy.misc.imread(img_file, mode='RGB')
        img = load_img(img_file, target_size=self.target_size)
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)

        img = vgg16.preprocess_input(img)
        datum = np.squeeze(img)

        shuffle = np.random.randint(0, len(self.style_files), 1)
        img_file = self.style_files[shuffle[0]]
        # img = scipy.misc.imread(img_file, mode='RGB')
        img = load_img(img_file, target_size=self.target_size)
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)

        img = vgg16.preprocess_input(img)
        style_img = np.squeeze(img)

        # datum = img / 255.
        return [datum, style_img], [datum, style_img, datum]

    def __getitem__(self, key):
        return self.get_example(key)


def load_data(file_name, index, batch_size):
    # Loading data
    pasacl_train_tmp = PascalVOC2012SegmentationDataset(file_name)
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
