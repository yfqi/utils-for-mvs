import numpy as np
import re
import sys
import argparse
import os
import cv2
from PIL import Image

class edge_crop:
    def __init__(self):
        self.image = None
        self.image_edge = None
        self.depth_pfm = None
        self.depth_png = None
        self.depth_edge = None
        self.depth_edge_dilate = None
        self.depth_edge_crop = None
        self.dim = [960,540]
    def read_img(self,image_path):
        image = cv2.imread(image_path,0)
        self.image = cv2.resize(image,self.dim, cv2.INTER_NEAREST)
    def read_pfm_depth(self,depth_pfm_path):
        self.depth_pfm = self.read_pfm(depth_pfm_path)
    def read_png_depth(self,depth_png_path):
        self.depth_png = cv2.imread(depth_png_path,flags=cv2.IMREAD_GRAYSCALE)
        #self.depth_png = cv2.resize(depth_png,self.dim, cv2.INTER_NEAREST)
    def pfm2png(self):
        min_depth = 2
        max_depth = 13
        depth = np.maximum(self.depth_pfm, min_depth)
        depth = np.minimum(depth, max_depth)
        self.depth_png = 255*(1.0/depth-1/max_depth)/(1/min_depth-1/max_depth)
        image = Image.fromarray(depth).convert("L")    
        #image.show()
        # image.save(pngfile)
    def get_image_edge(self):
        detected_edges = cv2.GaussianBlur(self.image,(3,3),0)
        edges = cv2.Canny(detected_edges,24,24*3)
        self.image_edge = cv2.resize(edges,self.dim,cv2.INTER_NEAREST)
        # cv2.imshow("img_edge",self.image_edge)
        # cv2.waitKey(0)
    def get_depth_edge(self):
        detected_edges = cv2.GaussianBlur(self.depth_png,(3,3),0)
        self.depth_edge = cv2.Canny(detected_edges,4,12)
    def get_depth_edge_dilate(self):
        kernel = np.ones((3,3), np.uint8)
        self.depth_edge_dilate  = cv2.dilate(self.depth_edge, kernel, iterations=1)
    def get_cropped_image_edge(self):
        self.depth_edge_crop = cv2.bitwise_and(self.image_edge, self.depth_edge_dilate)
    def save_edge(self,outfile):
        cv2.imwrite('img_edge.png', np.hstack((self.image, self.image_edge)))
        cv2.imwrite('depth_edge.png', np.hstack((self.depth_png,self.depth_edge_dilate)))
        cv2.imwrite('crop_edge.png', np.hstack((self.image_edge,self.depth_edge_crop)))
    def read_pfm(self,filename):
        file = open(filename, 'rb')
        color = None
        width = None
        height = None
        scale = None
        endian = None

        header = file.readline().decode('utf-8').rstrip()
        if header == 'PF':
            color = True
        elif header == 'Pf':
            color = False
        else:
            raise Exception('Not a PFM file.')

        dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception('Malformed PFM header.')

        scale = float(file.readline().rstrip())
        if scale < 0:  # little-endian
            endian = '<'
            scale = -scale
        else:
            endian = '>'  # big-endian

        data = np.fromfile(file, endian + 'f')
        shape = (height, width, 3) if color else (height, width)

        data = np.reshape(data, shape)
        #data = np.flipud(data)
        data = cv2.resize(data,[1920,1080],cv2.INTER_NEAREST)
        file.close()
        return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert depth .png to .png')

    parser.add_argument('-image', required=False, type=str, default="../data/images",help='imge file name *.png.')
    parser.add_argument('-depth_pfm', required=False, type=str, default="../data/gt/frame_251/", help='ground true depth file path.')
    parser.add_argument('-out',   required=False, type=str, default="../data/out/image_edge_crop",help='out file name.')
    parser.add_argument('-nums', type=int, default=1)
    args = parser.parse_args()
    for i in range(0, args.nums):
        img_file_name = os.path.join(args.image, "{}_p.png".format(i))
        depth_pfm_name = os.path.join(args.depth_pfm, "pfm/{}.pfm".format(i))
        depth_png_name = os.path.join(args.depth_pfm, "png/{:0>8}.png".format(i))
        out_depth_file_name = os.path.join(args.out, "{}.png".format(i))
        edge = edge_crop()
        edge.read_img(img_file_name)
        #edge.read_pfm_depth(depth_pfm_name)
        edge.read_png_depth(depth_png_name)
        #edge.pfm2png()
        edge.get_image_edge()
        edge.get_depth_edge()
        edge.get_depth_edge_dilate()
        edge.get_cropped_image_edge()
        edge.save_edge(out_depth_file_name)