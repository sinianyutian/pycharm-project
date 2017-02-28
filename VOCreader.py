# -*- coding: utf-8 -*-

import os
import shutil

basepath = "/home/andrew/VOC2012/"
respath = "/home/andrew/VOC-CLS/"
directory = basepath + "ImageSets/Main/"
imagedir = basepath + "JPEGImages/"
files = os.listdir(directory)


def parseLine(file):
    outimg = respath + "/" + file.replace("_trainval.txt",""+"/")
    try:
        os.mkdir(outimg)
    except:
        print(file + " already exists!")

    fh = open(directory + file)
    for line in fh.readlines():
        # image,tag=line.split(" ")[0:2]
        line = line.replace("\n", "").split(" ")
        image = line[0]
        tag = int(line[-1])
        if tag == 1:
            shutil.copy(imagedir + image+".jpg", outimg + image+".jpg")
            # print(imagedir+image+".jpg")
            # print(outimg+image+".jpg")
            # print(image, tag)
    return


try:
    os.mkdir(respath)
except:
    print(respath + " already exists!")

length = 0
for file in files:
    if file.endswith("_trainval.txt"):
        length += 1
        print(file)
        path = os.path.join(directory, file)
        print(path)
        parseLine(file)

print(length)
