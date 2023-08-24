# Open cv library
import cv2
# matplotlib for displaying the images 
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import random
import math
import numpy as np
import imutils
import os.path as path
import os
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--path", type=str)
parser.add_argument("--store", type=str)
parser.add_argument("--threshold", type=int, default=40)
parser.add_argument("--minCell", type=int, default=40)
parser.add_argument("--reshape", type=int, default=6000)
parser.add_argument("--renderavg", type=str)
options = parser.parse_args()


if options.path:
    print("path: {}".format(options.path))
    img_name = options.path
else:
    img_name = None
    print("Enter --path Argument for Image")

if options.store:
    print("store: {}".format(options.store))
    if options.store == "False":
        storebool = False
    else:
        storebool = True
else:
    storebool = False
    print("Enter --store False or True to store tiled images")

if options.renderavg:
    print("renderavg: {}".format(options.renderavg))
    if options.renderavg == "False":
        renderbool = False
    else:
        renderbool = True
else:
    renderbool = False

crop_img = cv2.imread(img_name)
imgT = imutils.resize(crop_img, width = options.reshape)
filedir, file_extension = os.path.splitext(img_name)

outpath = filedir + '/quadtree/'

class Node():
    def __init__(self, x0, y0, w, h):
        self.x0 = x0
        self.y0 = y0
        self.width = w
        self.height = h
        self.children = []

    def get_width(self):
        return self.width
    
    def get_height(self):
        return self.height
    
    def get_points(self):
        return self.points
    
    def get_points(self, img):
        return img[self.x0:self.x0 + self.get_width(), self.y0:self.y0+self.get_height()]
    
    def get_error(self, img):
        pixels = self.get_points(img)
        b_avg = np.mean(pixels[:,:,0])
        b_mse = np.square(np.subtract(pixels[:,:,0], b_avg)).mean()
    
        g_avg = np.mean(pixels[:,:,1])
        g_mse = np.square(np.subtract(pixels[:,:,1], g_avg)).mean()
        
        r_avg = np.mean(pixels[:,:,2])
        r_mse = np.square(np.subtract(pixels[:,:,2], r_avg)).mean()
        
        e = r_mse * 0.2989 + g_mse * 0.5870 + b_mse * 0.1140
        
        return (e * img.shape[0]* img.shape[1])/90000000


class QTree():
    def __init__(self, stdThreshold, minPixelSize, img):
        self.threshold = stdThreshold
        self.min_size = minPixelSize
        self.minPixelSize = minPixelSize
        self.img = img
        self.root = Node(0, 0, img.shape[0], img.shape[1])

    def get_points(self):
        return img[self.root.x0:self.root.x0 + self.root.get_width(), self.root.y0:self.root.y0+self.root.get_height()]
    
    def subdivide(self):
        recursive_subdivide(self.root, self.threshold, self.minPixelSize, self.img)
    
    def graph_tree(self):
        fig = plt.figure(figsize=(10, 10))
        plt.title("Quadtree")
        c = find_children(self.root)
        print("Number of segments: %d" %len(c))
        for n in c:
            plt.gcf().gca().add_patch(patches.Rectangle((n.y0, n.x0), n.height, n.width, fill=False))
        plt.gcf().gca().set_xlim(0,img.shape[1])
        plt.gcf().gca().set_ylim(img.shape[0], 0)
        plt.axis('equal')
        plt.show()
        return

    def render_img(self, outpath, thickness = 1, color = (0,0,255)):
        imgc = self.img.copy()
        c = find_children(self.root)
        counter = 0

        name = img_name.split('/')[-1]
        for n in c:
            pixels = n.get_points(self.img)
            # grb
            gAvg = math.floor(np.mean(pixels[:,:,0]))
            rAvg = math.floor(np.mean(pixels[:,:,1]))
            bAvg = math.floor(np.mean(pixels[:,:,2]))

            if renderbool:
              imgc[n.x0:n.x0 + n.get_width(), n.y0:n.y0+n.get_height(), 0] = gAvg
              imgc[n.x0:n.x0 + n.get_width(), n.y0:n.y0+n.get_height(), 1] = rAvg
              imgc[n.x0:n.x0 + n.get_width(), n.y0:n.y0+n.get_height(), 2] = bAvg
            counter += 1
            #print (str(n))

            """
            if file exists then delete and make new one
            """

            with open(filedir + '/' + str(name[0:10]) + '_quadNodes_Storage.txt', 'a') as f:
                #f.write('n.x0:' + str(n.x0) +' , ' + "n.x0+n.get_width():" + str(n.x0 + n.get_width()) + ' , ' + 'n.y0:' + str(n.y0) + ' , ' + "n.y0+n.get_height():" + str(n.y0 + n.get_height()))
                f.write(str(n.x0) +' , ' + str(n.x0 + n.get_width()) + ' , ' + str(n.y0) + ' , ' + str(n.y0 + n.get_height()))
                f.write('\n')
            
            if storebool and not renderbool:
                cv2.imwrite(os.path.join(outpath, str(1000000+counter)+'test.jpg'),imgc[n.x0:n.x0 + n.get_width(), n.y0:n.y0+n.get_height()])

        if thickness > 0 and not renderbool:
            for n in c:
                # Draw a rectangle
                imgc = cv2.rectangle(imgc, (n.y0, n.x0), (n.y0+n.get_height(), n.x0+n.get_width()), color, thickness)
        return imgc

def recursive_subdivide(node, k, minPixelSize, img):

    if node.get_error(img)<=k:
        return
    w_1 = int(math.floor(node.width/2))
    w_2 = int(math.ceil(node.width/2))
    h_1 = int(math.floor(node.height/2))
    h_2 = int(math.ceil(node.height/2))


    if w_1 <= minPixelSize or h_1 <= minPixelSize:
        return
    x1 = Node(node.x0, node.y0, w_1, h_1) # top left
    recursive_subdivide(x1, k, minPixelSize, img)

    x2 = Node(node.x0, node.y0+h_1, w_1, h_2) # btm left
    recursive_subdivide(x2, k, minPixelSize, img)

    x3 = Node(node.x0 + w_1, node.y0, w_2, h_1)# top right
    recursive_subdivide(x3, k, minPixelSize, img)

    x4 = Node(node.x0+w_1, node.y0+h_1, w_2, h_2) # btm right
    recursive_subdivide(x4, k, minPixelSize, img)

    node.children = [x1, x2, x3, x4]
   

def find_children(node):
   if not node.children:
       return [node]
   else:
       children = []
       for child in node.children:
           children += (find_children(child))
   return children

def displayQuadTree(imgT, threshold=4, minCell=20, img_boarder=20, line_boarder=1, line_color=(0,0,255)):
    #imgT= cv2.imread(img_name)

    #w = (imgT.shape[0]*2)/3
    #h = imgT.shape[0]
    #center = (imgT.shape[0] / 2, imgT.shape[1] / 2)
    #x = center[1] - w/2
    #y = center[0] - h/2
    #crop_img = imgT[int(y):int(y+h), int(x):int(x+w)]


    if not os.path.exists(outpath):
        # Create a new directory because it does not exist 
        os.makedirs(outpath)
    else:
        for myfile in os.listdir(outpath):
            os.remove(os.path.join(outpath,myfile))

    name = img_name.split('/')[-1]
    print (name)
    file_name = "output_" + name

    #cleantextfile
    txtfile = filedir + '/' + str(name[0:10]) + '_quadNodes_Storage.txt'
    print (txtfile)
    if os.path.exists(txtfile):
      os.remove(txtfile)

    qt = QTree(threshold, minCell, imgT) 
    qt.subdivide()
    qtImg= qt.render_img(outpath, thickness=line_boarder, color=line_color)
    
    cv2.imwrite(filedir + '/' + file_name,qtImg)

displayQuadTree(imgT, threshold=options.threshold, minCell=options.minCell, line_color=(0,0,255))




"""
Save out each subdivision overlay by extracting from the original image 
and saving to separate file in directory.
This will save out images of different sizes that can be resized and compared
with LPIPS, shuffled and sorted to reconstitute new images
"""
