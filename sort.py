import argparse
import numpy as np
import os
import imutils
import cv2
import random
import shutil
from pathlib import Path
import tensorflow as tf
import torch.nn.functional as F
#from matplotlib import pyplot as plt

def parse_args():
	desc = "Tools to normalize an image dataset" 
	parser = argparse.ArgumentParser(description=desc)

	parser.add_argument('-v','--verbose', action='store_true',
		help='Print progress to console.')

	parser.add_argument('-i','--input_folder', type=str,
		default='./input/',
		help='Directory path to the inputs folder. (default: %(default)s)')

	parser.add_argument('-o','--output_folder', type=str,
		default='./output/',
		help='Directory path to the outputs folder. (default: %(default)s)')

	parser.add_argument('-p','--process_type', type=str,
		default='exclude',
		help='Process to use. ["exclude","sort","tagsort","lpips","channels"] (default: %(default)s)')

	parser.add_argument('-n','--network', type=str,
		default='alex',
		help='Network to use for the LPIPS sort process. Options: alex, vgg, squeeze (default: %(default)s)')

	parser.add_argument('-f','--file_extension', type=str,
		default='png',
		help='file type ["png","jpg"] (default: %(default)s)')

	parser.add_argument('--start_img', type=str,
		help='image for comparison (for lpips process)')

	parser.add_argument('--use_gpu', action='store_true', 
		help='use GPU (for lpips process)')

	args = parser.parse_args()
	return args

def main():
	global args
	global count
	global inter
	args = parse_args()
	count = int(0)
	inter = cv2.INTER_CUBIC
	os.environ['OPENCV_IO_ENABLE_JASPER']= "true"

	if os.path.isdir(args.input_folder):
		print("Processing folder: " + args.input_folder)
	else:
		print("Not a working input_folder path: " + args.input_folder)
		return;

	# sort using LPIPS
	if(args.process_type == "lpips"):
		txtfile = 'sorting_INDEX.txt'
		if os.path.exists(txtfile):
			os.remove(txtfile)

		import lpips

		loss_fn = lpips.LPIPS(net=args.network,version='0.1')	

		if os.path.isdir(args.input_folder):
			inputfiles = os.listdir(args.input_folder)

			img0 = lpips.im2tensor(lpips.load_image(args.start_img))

			if not os.path.exists(args.output_folder):
				os.makedirs(args.output_folder)
			else:
				for myfile in os.listdir(args.output_folder):
					os.remove(os.path.join(args.output_folder,myfile))

			if(args.use_gpu):
				loss_fn.cuda()
				img0 = img0.cuda()

			for filename in inputfiles:
				file_path = os.path.join(args.input_folder, filename)
				img1 = lpips.im2tensor(lpips.load_image(file_path))
				img1 = F.interpolate(img1, size=(tf.shape(img0.cpu())[2],tf.shape(img0.cpu())[3]))

				if(args.use_gpu):
					img1 = img1.cuda()

				dist01 = loss_fn.forward(img0,img1)
				if(args.verbose): print('%s Distance: %.3f'%(filename,dist01))

				with open(txtfile, 'a') as f:
					f.write(str('%.3f'%(dist01)) +' , ')
					f.write('\n')

				new_path = os.path.join(args.output_folder, '%.3f'%(dist01) + '.png')
				outimg = cv2.imread(file_path)
				cv2.imwrite(new_path, outimg)
	return


if __name__ == "__main__":
	main()