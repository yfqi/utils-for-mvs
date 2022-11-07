import cv2
import numpy as np
import re
import cv2
import numpy as np
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




def CannyThreshold(lowThreshold):
    detected_edges = cv2.GaussianBlur(depth_hr, (3, 3), 0)
    detected_edges = cv2.Canny(detected_edges, lowThreshold, lowThreshold * ratio, apertureSize=kernel_size)
    #dst = cv2.bitwise_and(depth_hr, depth_hr, mask=detected_edges)  # just add some colours to edges from original image.
    print(lowThreshold)
    if not os.path.exists(path+'/'+dir):
        os.makedirs(path+'/'+dir)
    cv2.imwrite(path+'/'+dir+'/'+file +'.png', detected_edges)



lowThreshold = 0
max_lowThreshold = 100
ratio = 3
kernel_size = 3

#img = cv2.imread('rect_001_0_r5000.png')
#gray = cv2.cvtColor(depth_hr, cv2.COLOR_BGR2GRAY)

#cv2.namedWindow('canny demo')

#cv2.createTrackbar
photo_path = '/mnt2/yfq/CasMVSNet/mvs_training/dtu/Depths_raw'
path = '/mnt2/yfq/CasMVSNet/mvs_training/dtu/depthmap_tdt/30'

for root, dirs, _ in os.walk(photo_path):
    for dir in dirs:
        for root1, _, files in os.walk(root +"/"+ dir):
            for file in files:
                if file.endswith('pfm'):
                    filename = photo_path + '/' + dir + '/' + file
                    depth_hr = np.array(read_pfm(filename)[0], dtype=np.uint8)
                    CannyThreshold(30)  # initialization
                    '''if cv2.waitKey(0) == 27:
                        cv2.destroyAllWindows()'''




