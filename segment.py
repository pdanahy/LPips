from __future__ import print_function
import cv2 as cv
import imutils
import numpy as np
import argparse
import random as rng
import os
rng.seed(12345)
parser = argparse.ArgumentParser(description='Code for Image Segmentation with Distance Transform and Watershed Algorithm.\
    Sample code showing how to segment overlapping objects using Laplacian filtering, \
    in addition to Watershed and Distance Transformation')
parser.add_argument('--input', help='Path to input image.', default='cards.png')
parser.add_argument('--threshold', help='Threshold for Distance Calculation', default='0.4')
args = parser.parse_args()



def segmentation_call(filepath, count):
  src = cv.imread(filepath)

  src_shape = src.shape

  if src is None:
      print('Could not open or find the image:', filepath)
      exit(0)
  src = imutils.resize(src, width = 600)
  src[np.all(src == 255, axis=2)] = 255
  kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float32)
  imgLaplacian = cv.filter2D(src, cv.CV_32F, kernel)
  sharp = np.float32(src)
  imgResult = sharp - imgLaplacian
  imgResult = np.clip(imgResult, 0, 255)
  imgResult = imgResult.astype('uint8')
  imgLaplacian = np.clip(imgLaplacian, 0, 255)
  imgLaplacian = np.uint8(imgLaplacian)
  bw = cv.cvtColor(imgResult, cv.COLOR_BGR2GRAY)
  _, bw = cv.threshold(bw, 40, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
  #cv.imwrite('threshold.jpg',bw)
  dist = cv.distanceTransform(bw, cv.DIST_L2, 3)
  cv.normalize(dist, dist, 0, 1.0, cv.NORM_MINMAX)
  _, dist = cv.threshold(dist, float(args.threshold), 1.0, cv.THRESH_BINARY)
  kernel1 = np.ones((3,3), dtype=np.uint8)
  dist = cv.dilate(dist, kernel1)
  dist_8u = dist.astype('uint8')
  # Find total markers
  contours,_ = cv.findContours(dist_8u, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
  # Create the marker image for the watershed algorithm
  markers = np.zeros(dist.shape, dtype=np.int32)
  # Draw the foreground markers
  for i in range(len(contours)):
      cv.drawContours(markers, contours, i, (i+1), -1)
  # Draw the background marker
  cv.circle(markers, (5,5), 1, (255,255,255), -1)
  markers_8u = (markers * 10).astype('uint8')
  cv.watershed(imgResult, markers)
  #mark = np.zeros(markers.shape, dtype=np.uint8)
  mark = markers.astype('uint8')
  mark = cv.bitwise_not(mark)
  # uncomment this if you want to see how the mark
  # image looks like at that point
  #cv.imshow('Markers_v2', mark)
  # Generate random colors
  colors = []
  for contour in contours:
      colors.append((rng.randint(0,256), rng.randint(0,256), rng.randint(0,256)))
  # Create the result image
  dst = np.zeros((markers.shape[0], markers.shape[1], 3), dtype=np.uint8)
  # Fill labeled objects with random colors
  for i in range(markers.shape[0]):
      for j in range(markers.shape[1]):
          index = markers[i,j]
          if index > 0 and index <= len(contours):
              dst[i,j,:] = colors[index-1]
  # Visualize the final image
  dst = imutils.resize(dst, width = src_shape[1])
  outpath = os.path.split(filepath)[1]
  cv.imwrite(os.path.join(path, outpath), dst)
  print("done")
















path = os.path.splitext(args.input)[0]
path = os.path.join(path, 'segmented')

if not os.path.exists(path):
    os.makedirs(path)

count = 0

if os.path.isdir(args.input):
  for thisfile in os.listdir(args.input):
    count += 1
    segmentation_call(os.path.join(args.input,thisfile), count)
else:
  segmentation_call(args.input, count)
