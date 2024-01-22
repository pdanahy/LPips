import argparse
import os
import lpips
import tensorflow as tf
import numpy as np
#device_name = tf.test.gpu_device_name()
#if device_name != '/device:GPU:0':
#  raise SystemError('GPU device not found')
#print('Found GPU at: {}'.format(device_name))
import cv2
import imutils
from skimage.metrics import structural_similarity as compare_ssim
import torch.nn.functional as F
import time
import torchvision

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-d0','--dir0', type=str, default='./imgs/ex_dir0')
parser.add_argument('-d1','--dir1', type=str, default='./imgs/ex_dir1')
parser.add_argument('-o','--out', type=str, default='./imgs/example_dists.txt')
parser.add_argument('-v','--version', type=str, default='0.1')
parser.add_argument('-threed','--threed', type=str, default='0')
parser.add_argument('--use_gpu', action='store_true', help='turn on flag to use GPU')
parser.add_argument('-qtfixed','--qtfixed', type=str, default='False')
parser.add_argument('--continue_run', type=str, default='False')

opt = parser.parse_args()



## Initializing the model
loss_fn = lpips.LPIPS(net='alex',version=opt.version)
if(opt.use_gpu):
    loss_fn.cuda()

# crawl directories
#files = os.listdir(opt.dir0)
Directory = opt.dir0

name, ext = os.path.splitext(os.path.split(opt.dir0)[1])
outpath = Directory + '_substituted_'

if not os.path.exists(outpath):
    # Create a new directory because it does not exist 
    os.makedirs(outpath)
elif opt.continue_run == 'True':
    pass
else:
    for myfile in os.listdir(outpath):
        os.remove(os.path.join(outpath,myfile))

if opt.threed == '1':
    txtfile = os.path.join(name + '/substituted_','lpipsIndex.txt')
    if os.path.exists(txtfile):
        os.remove(txtfile)

print( "\n------- RUNNING IMAGE LOOPS --------")
print("Find Files Here:" + str(outpath) + "\n")
total_start = time.time()

if opt.continue_run == 'True':
        listlength = len(os.listdir(os.path.join(name + '/substituted_','quadtree')))

list_of_files = sorted( filter( lambda x: os.path.isfile(os.path.join(Directory, x)),
                        os.listdir(Directory) ) )

for i,file in enumerate(list_of_files):
  start = time.time()
  print(file)
  if opt.continue_run == 'True':
      if i < listlength:
          print('')
      else:
        if(os.path.exists(os.path.join(opt.dir0,file))):
          
          # Load images
          img0 = lpips.im2tensor(lpips.load_image(os.path.join(opt.dir0,file))) # RGB image from [-1,1]

          clstimg = np.zeros((img0.shape[1],img0.shape[0],3), np.uint8)
          clstdist = 99999999999
          clstindex = None

          Directory2 = opt.dir1
          list_of_files2 = sorted( filter( lambda x: os.path.isfile(os.path.join(Directory2, x)),
                        os.listdir(Directory2) ) )
          for j,file2 in enumerate(list_of_files2):

            img1 = lpips.im2tensor(lpips.load_image(os.path.join(opt.dir1,file2)))
            
            if opt.qtfixed == 'False':
              if str(tf.shape(img1.cpu())) != str(tf.shape(img0.cpu())):
                img1 = F.interpolate(img1, size=(tf.shape(img0.cpu())[2],tf.shape(img0.cpu())[3]))

              #print(tf.shape(img1))

              if(opt.use_gpu):
                img0 = img0.cuda()
                img1 = img1.cuda()

              # Compute distance
              dist01 = loss_fn.forward(img0,img1)
              if dist01 < clstdist:
                clstdist = dist01
                clstimg = cv2.imread(os.path.join(opt.dir1,file2))
                #clstimg = os.path.join(opt.dir1,file2)
                if opt.threed == '1':
                  clstindex = j
            else:
              #print (abs(int(tf.shape(img1.cpu())[2]) - int(tf.shape(img0.cpu())[2])))
              if abs(int(tf.shape(img1.cpu())[2]) - int(tf.shape(img0.cpu())[2])) < 10:
                img1 = F.interpolate(img1, size=(tf.shape(img0.cpu())[2],tf.shape(img0.cpu())[3]))
                if(opt.use_gpu):
                  img0 = img0.cuda()
                  img1 = img1.cuda()

                # Compute distance
                dist01 = loss_fn.forward(img0,img1)
                if dist01 < clstdist:
                  clstdist = dist01
                  #clstimg = os.path.join(opt.dir1,file2)
                  clstimg = cv2.imread(os.path.join(opt.dir1,file2))
                  clstindex = j
              else:
                if opt.threed == '1':
                  clstindex = 0
                else:
                  pass
                #clstindex = None
                #clstimg = np.zeros((img0.shape[1],img0.shape[0],3), np.uint8)

          if opt.threed == '1':
            with open(txtfile, 'a') as f:
              f.write(str(clstindex) +' , ')
              f.write('\n')


          cv2.imwrite((os.path.join(outpath, file)), clstimg)
  else:
    if(os.path.exists(os.path.join(opt.dir0,file))):
      
      # Load images
      img0 = lpips.im2tensor(lpips.load_image(os.path.join(opt.dir0,file))) # RGB image from [-1,1]
      cvimg0 = cv2.imread(os.path.join(opt.dir0,file))
      img0height, img0width, channels = cvimg0.shape

      clstimg = np.zeros((img0.shape[1],img0.shape[0],3), np.uint8)
      clstdist = 99999999999
      clstindex = None

      
      Directory2 = opt.dir1
      list_of_files2 = sorted( filter( lambda x: os.path.isfile(os.path.join(Directory2, x)),
                    os.listdir(Directory2) ) )
      for j,file2 in enumerate(list_of_files2):

        img1 = lpips.im2tensor(lpips.load_image(os.path.join(opt.dir1,file2)))
        
        if opt.qtfixed == 'False':
          if str(tf.shape(img1.cpu())) != str(tf.shape(img0.cpu())):
            #img1 = F.interpolate(img1, size=(tf.shape(img0.cpu())[2],tf.shape(img0.cpu())[3]))
            img1 = F.interpolate(img1, size=(img0height,img0width))
          
          img0 = torchvision.transforms.functional.adjust_saturation(img0, 0.1) 
          img1 = torchvision.transforms.functional.adjust_saturation(img1, 0.1) 

          if(opt.use_gpu):
            img0 = img0.cuda()
            img1 = img1.cuda()

          # Compute distance
          dist01 = loss_fn.forward(img0,img1)
          if dist01 < clstdist:
            clstdist = dist01
            clstimg = cv2.imread(os.path.join(opt.dir1,file2))
            #clstimg = os.path.join(opt.dir1,file2)
            if opt.threed == '1':
              clstindex = j
        else:
          #print (abs(int(tf.shape(img1.cpu())[2]) - int(tf.shape(img0.cpu())[2])))
          if abs(int(tf.shape(img1.cpu())[2]) - int(tf.shape(img0.cpu())[2])) < 10:
            img1 = F.interpolate(img1, size=(tf.shape(img0.cpu())[2],tf.shape(img0.cpu())[3]))
            if(opt.use_gpu):
              img0 = img0.cuda()
              img1 = img1.cuda()

            # Compute distance
            dist01 = loss_fn.forward(img0,img1)
            if dist01 < clstdist:
              clstdist = dist01
              #clstimg = os.path.join(opt.dir1,file2)
              clstimg = cv2.imread(os.path.join(opt.dir1,file2))
              clstindex = j
          else:
            pass
            #clstindex = None
            #clstimg = np.zeros((img0.shape[1],img0.shape[0],3), np.uint8)

      if opt.threed == '1':
        with open(txtfile, 'a') as f:
          f.write(str(clstindex) +' , ')
          f.write('\n')


      cv2.imwrite((os.path.join(outpath, file)), clstimg)
  
  end = time.time()
  print("Elapsed Time per Image:" + str(end - start))
total_end = time.time()
print("Total Elapsed Time:" + str(total_end - total_start))
