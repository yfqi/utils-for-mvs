import cv2
import time
import matplotlib.pyplot as plt
import segmentation_refinement as refine
import numpy as np
import os

pic_dir = '/mnt2/yfq/datasets/oly/Rectified'

refiner = refine.Refiner(device='cuda:0')  # device can also be 'cpu'
for root, dirs, files in os.walk(pic_dir):
    for dir in dirs:
        for root1,_,files1 in (root+dir+"/images"):
            for file1 in files1:
                image = cv2.imread(root+file1)
                name = os.path.splitext(file1)[0]
                mask = cv2.imread('/mnt2/yfq/datasets/oly/wlt/27'+'/'+dir+'/'+file1, cv2.IMREAD_GRAYSCALE)

        #mask=np.load('in/scan1/predictions/{}.txt.npy'.format(file),encoding = "latin1")
# model_path can also be specified here
# This step takes some time to load the model

        #mask = mask.squeeze(0)
# Fast - Global step only.
# Smaller L -> Less memory usage; faster in fast mode.
                output = refiner.refine(image, mask, fast=False, L=900)

# this line to save output
                print(0)
                cv2.imwrite('out/out_{}.png'.format(name), output)
                print(1)

"""        plt.imshow(output)
        
        plt.show()"""