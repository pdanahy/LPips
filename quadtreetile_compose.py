# Open cv library
import cv2
# matplotlib for displaying the images 
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import random
import math
import numpy as np
import imutils
import os
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--path", type=str)
parser.add_argument("--ref", type=str)
parser.add_argument("--reshape", type=int, default=6000)
options = parser.parse_args()

if options.path:
    print("path: {}".format(options.path))
    Directory = options.path
else:
    Directory = None
    print("Enter --path Argument for Directory")

if options.ref:
    print("ref: {}".format(options.ref))
    img_name = options.ref
else:
    img_name = None
    print("Enter --ref Argument for Image Reference")

path = os.path.splitext(img_name)[0]
print (path)
name = path.split('/')[-1]
name = os.path.join(path, str(name[0:10]))

cvimg1 = cv2.imread(img_name)

#w = (cvimg1.shape[0]*2)/3
#h = cvimg1.shape[0]
#center = (cvimg1.shape[0] / 2, cvimg1.shape[1] / 2)
#x = center[1] - w/2
#y = center[0] - h/2
#crop_img = cvimg1[int(y):int(y+h), int(x):int(x+w)]


imgT = imutils.resize(cvimg1, width = options.reshape)
#imgT = imutils.resize(cvimg1, width = options.reshape)

# make blank numpy array the shape of the reference image
blankimg = np.zeros(shape=(imgT.shape))


# Loop through nodes and images and fill blank array based on node properties (x0:x1,y0:y1)


with open(name + '_quadNodes_Storage.txt') as file:
    lines = file.readlines()
    lines = [line.rstrip() for line in lines]

list_of_files = sorted( filter( lambda x: os.path.isfile(os.path.join(Directory, x)),
                        os.listdir(Directory) ) )

for i,file in enumerate(list_of_files):
    #if i < len(lines):
    cvimg2 = cv2.imread(os.path.join(Directory, file))
    
    nodeprops = lines[i].split(", ")
    cvimg2 = cv2.resize(cvimg2, ((int(nodeprops[3])-int(nodeprops[2])),int(nodeprops[1])-int(nodeprops[0])))
    #randomnum = random.randrange(1, 11, 2)
    #cvimg2 = cv2.GaussianBlur(cvimg2,(randomnum,randomnum),0)

    #if cvimg2.shape == (int(nodeprops[1])-int(nodeprops[0]),int(nodeprops[3])-int(nodeprops[2]), 3):
    blankimg[int(nodeprops[0]):int(nodeprops[1]), int(nodeprops[2]):int(nodeprops[3])] = cvimg2

cv2.imwrite(name + '_quadNodes_Storage.jpg', blankimg)

    
