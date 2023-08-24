import cv2
import os
import numpy as np
import imutils
from argparse import ArgumentParser


def split(img, window_size, filepath):
    print ("GO")
    full_path = os.path.join(filepath,'tiled_'+str(window_size))
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    img = cv2.imread(img)
    img = imutils.resize(img, width=2600)
    sh = list(img.shape)

    stride = window_size

    nrows, ncols = img.shape[0] // window_size, img.shape[1] // window_size
    splitted = []
    for i in range(nrows):
        for j in range(ncols):
            h_start = j*stride
            v_start = i*stride
            cropped = img[v_start:v_start+stride, h_start:h_start+stride]
            splitted.append(cropped)

            #root, filename = os.path.split(filepath)
            outpath = os.path.join(full_path,'tiled_' + str(1000+i) + '_' + str(1000+j) + '.jpg')
            print (outpath)
            cv2.imwrite(outpath, cropped)
    return splitted


cwd = os.getcwd()
parser = ArgumentParser()
parser.add_argument("--path", type=str)
parser.add_argument("--window", type=int)
options = parser.parse_args()

if options.path:
    print("path: {}".format(options.path))
    directory = options.path
else:
    directory = None
    print("Enter --path Argument for Directory")

if options.window:
    print("window: {}".format(options.window))
    window_size = options.window
else:
    window_size = 20

if directory != None:
    if os.path.isdir(directory):
        files = os.listdir(directory)
        for file in files:
            name, ext = os.path.splitext(file)
            if ext == '.jpg' or ext == '.png' or ext == '.jpeg':
                out = split(file, window_size, name)
    elif os.path.isfile(directory):
        name, ext = os.path.splitext(directory)
        if ext == '.jpg' or ext == '.png' or ext == '.jpeg':
            out = split(directory, window_size, name)