import os.path as osp

import numpy as np
import scipy.misc

from keras.applications import vgg16
from keras.preprocessing import image
from keras.preprocessing.image import load_img

from segmentation_dataset import SegmentationDatasetBase


class PascalVOC2012SegmentationDatasetImagePair(SegmentationDatasetBase):
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

        return img

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
