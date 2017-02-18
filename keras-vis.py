# -*- coding: utf-8 -*-
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

model = VGG16(weights='imagenet')
model.summary()

img_path = 'images/tiger.JPEG'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
tags = decode_predictions(preds, top=3)[0]
# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
print('Predicted:', decode_predictions(preds, top=3)[0])

layer_name = 'predictions'
layer_idx = [idx for idx, layer in enumerate(model.layers) if layer.name == layer_name][0]

import cv2
import numpy as np

from vis.utils import utils
# from vis.visualization import visualize_saliency
from vis.visualization import visualize_cam

seed_img = utils.load_img(img_path, target_size=(224, 224))
pred_class = np.argmax(preds)
# heatmap = visualize_saliency(model, layer_idx, [pred_class], seed_img=seed_img)
heatmap = visualize_cam(model, layer_idx, [pred_class], seed_img=seed_img)
cv2.imshow('Attention - {}'.format(tags[1]), heatmap)
cv2.waitKey(0)


# from keras.applications import vgg16
# from vis.utils.vggnet import VGG16
# from vis.optimizer import Optimizer
# from vis.losses import ActivationMaximization
# from vis.regularizers import TotalVariation, LPNorm


# Build the VGG16 network with ImageNet weights
# model = VGG16(weights='imagenet', include_top=True)
# model = vgg16.VGG16(weights='imagenet', include_top=True)
# print('Model loaded.')
#
# # The name of the layer we want to visualize
# # (see model definition in vggnet.py)
# layer_name = 'predictions'
# layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])
# output_class = [20]
#
# losses = [
#     (ActivationMaximization(layer_dict[layer_name], output_class), 2),
#     (LPNorm(model.input), 10),
#     (TotalVariation(model.input), 10)
# ]
# opt = Optimizer(model.input, losses)
#
# opt.minimize(max_iter=500, verbose=True,
#              progress_gif_path='opt_progress')