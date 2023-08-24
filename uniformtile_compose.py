import cv2
import os
import numpy as np
import imutils
from argparse import ArgumentParser
from pathlib import Path


def vconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    #w_min = min(im.shape[1] for im in im_list)
    w_min = rheight
    im_list_resize = [cv2.resize(im, (w_min, int(im.shape[0] * w_min / im.shape[1])), interpolation=interpolation)
                      for im in im_list]
    return cv2.vconcat(im_list_resize)

def hconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    #h_min = min(im.shape[0] for im in im_list)
    h_min = rwidth
    im_list_resize = [cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation)
                      for im in im_list]
    return cv2.hconcat(im_list_resize)

def concat_tile(im_list_2d):
    return cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in im_list_2d])

def concat_tile_resize(im_list_2d, interpolation=cv2.INTER_CUBIC):
    im_list_v = [hconcat_resize_min(im_list_h, interpolation=cv2.INTER_CUBIC) for im_list_h in im_list_2d]
    return vconcat_resize_min(im_list_v, interpolation=cv2.INTER_CUBIC)



parser = ArgumentParser()
parser.add_argument("--path", type=str)
parser.add_argument("--ref", type=str)
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

if options.ref:
    print("ref: {}".format(options.ref))
    refimg = options.ref
else:
    refimg = None
    print("Enter --ref Argument for Reference Image")

refimg = cv2.imread(refimg)
refimg = imutils.resize(refimg, width=2600)
print(refimg.shape[0],refimg.shape[1])
rheight = refimg.shape[0]
rwidth = refimg.shape[1]
hstep = int(refimg.shape[1]/window_size)
hstep = hstep
print (hstep)
#wstep = refimg.shape[1]/window_size
nested_tile_list = []

if directory != None:
    if os.path.isdir(directory):
        files = os.listdir(directory)
        temp_list = []
        for i,file in enumerate(files):
            name, ext = os.path.splitext(file)
            if ext == '.jpg':
                temp_list.append(cv2.imread(os.path.join(directory,file)))
                if (i+1) % (hstep) == 0:
                    nested_tile_list.append(temp_list)
                    temp_list = []


im_tile_resize = concat_tile_resize(nested_tile_list)
#p = Path(directory)
cv2.imwrite(os.path.join('/content/drive/MyDrive/LPips/', 'test_opencv_concat_tile_resize_' + str(window_size) +'.jpg'), im_tile_resize)