import cv2
import numpy as np
import re
import cv2
import numpy as np
from PIL import Image
import os
def read_pfm(filename):
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
    data = np.flipud(data)
    file.close()
    return data, scale


##############################################################################################3

# a = features[0][0, 0, :]
# a = Image.fromarray((a * 255).cpu().numpy()).convert("L")
# a.save("./a.png")
# img1 = cv2.imread("./a.png")
# img1_c = cv2.applyColorMap(img1, cv2.COLORMAP_JET)
# cv2.imwrite("0a.png", img1_c)
#
# a = features[1][0, 0, :]
# a = Image.fromarray((a * 255).cpu().numpy()).convert("L")
# a.save("./a.png")
# img1 = cv2.imread("./a.png")
# img1_c = cv2.applyColorMap(img1, cv2.COLORMAP_JET)
# cv2.imwrite("1a.png", img1_c)
#
# a = features[2][0, 0, :]
# a = Image.fromarray((a * 255).cpu().numpy()).convert("L")
# a.save("./a.png")
# img1 = cv2.imread("./a.png")
# img1_c = cv2.applyColorMap(img1, cv2.COLORMAP_JET)
# cv2.imwrite("2a.png", img1_c)
#
# a = features[3][0, 0, :]
# a = Image.fromarray((a * 255).cpu().numpy()).convert("L")
# a.save("./a.png")
# img1 = cv2.imread("./a.png")
# img1_c = cv2.applyColorMap(img1, cv2.COLORMAP_JET)
# cv2.imwrite("3a.png", img1_c)
#
# a = features[4][0, 0, :]
# a = Image.fromarray((a * 255).cpu().numpy()).convert("L")
# a.save("./a.png")
# img1 = cv2.imread("./a.png")
# img1_c = cv2.applyColorMap(img1, cv2.COLORMAP_JET)
# cv2.imwrite("4a.png", img1_c)
#
# a = features[5][0, 0, :]
# a = Image.fromarray((a * 255).cpu().numpy()).convert("L")
# a.save("./a.png")
# img1 = cv2.imread("./a.png")
# img1_c = cv2.applyColorMap(img1, cv2.COLORMAP_JET)
# cv2.imwrite("5a.png", img1_c)

###############################################################################################
#img = cv2.imread('rect_001_0_r5000.png')
#gray = cv2.cvtColor(depth_hr, cv2.COLOR_BGR2GRAY)

#cv2.namedWindow('canny demo')

#cv2.createTrackbar



#filename1 = r'C:\Users\fqyu\Desktop\00000025-p.pfm'
filename = r'D:\原始素材\13\tdt.pfm'
#filename1 = r'C:\Users\fqyu\Desktop\eccv2022\10\11_new.pfm'
#filename2 = r'C:\Users\fqyu\Desktop\eccv2022\10\84_new.pfm'
#filename3 = r'C:\Users\fqyu\Desktop\eccv2022\10\8844.pfm'
depth_hr = np.array(read_pfm(filename)[0])
#depth_hr1 = np.array(read_pfm(filename1)[0])
# depth_hr2 = np.array(read_pfm(filename2)[0])
# depth_hr3 = abs(depth_hr2 - depth_hr)
# depth_hr4 = np.array(read_pfm(filename3)[0])


#depth_hr = np.array(read_pfm(filename)[0], dtype=np.uint8) #深度图
#depth_hr = np.array(read_pfm(filename)[0] * 255, dtype=np.uint8) #置信度图
#depth_hrp = np.array(read_pfm(filename1)[0], dtype=np.uint8)
#a = np.sum(depth_hrp-depth_hr)
# depth_png = np.array(Image.open(filename), dtype=int)

'''depth_image = (depth_png / 256.).astype(np.float32)
depth = np.expand_dims(depth_image, axis=2)
img0 = cv2.imread(filename)
gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
detected_edges = cv2.GaussianBlur(gray, (3, 3), 0)
tdt = cv2.Canny(detected_edges, 600, 25 * 3, apertureSize=3)

ret, binary = cv2.threshold(tdt, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
                #cv2.imshow("binary", binary)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
tdt = cv2.dilate(binary, kernel)'''

# depth_png[205:210,36:40,0:3] = 255

# dep=Image.fromarray((depth_hr-425)/3).convert("L")
dep=Image.fromarray((depth_hr/3)).convert("L")
# print(depth31)
dep.save("./depth.png")
img1 = cv2.imread("./depth.png")
img1_c = cv2.applyColorMap(img1,cv2.COLORMAP_JET)
cv2.imwrite("tdt.png",img1_c)
# img_c = cv2.applyColorMap(dep,cv2.COLORMAP_JET)
# #cv2.imwrite(r"C:\Users\fqyu\Desktop\r.png",img_c)
# cv2.imwrite(r'C:\Users\fqyu\Desktop\123.png',dep)







































