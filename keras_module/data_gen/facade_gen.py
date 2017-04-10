import numpy as np

try:
    from PIL import Image as pil_image
except ImportError:
    pil_image = None


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
