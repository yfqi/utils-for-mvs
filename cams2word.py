# -*- coding: utf-8 -*
import argparse
import numpy as np
import os

def read_cam_file(filename):
    with open(filename) as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
    # extrinsics: line [1,5), 4x4 matrix
    extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float64, sep=' ').reshape((4, 4))
    # intrinsics: line [7-10), 3x3 matrix
    intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float64, sep=' ').reshape((3, 3))
    # depth_min & depth_interval: line 11
    depth = np.fromstring(lines[11], dtype=np.float64, sep=' ').reshape((1, 4))
    return intrinsics, extrinsics, depth

def get_cams(datapath,view_ids = 40):
    proj_matrices = []
    H = [[ 0.92773339, -0.0818465,   0.36415917, -3.15122013],
    [-0.37267067, -0.25716154,   0.89161904,  0.02228677],
    [ 0.02067184, -0.9628962,  -0.26907914,  1.24433872],
    [ 0. ,         0. ,         0.,          1.        ]]
    for i in range(view_ids):
        proj_mat_filename = os.path.join(datapath, '{:0>8}_cam.txt').format(i)
        intrinsics, extrinsics, depth = read_cam_file(proj_mat_filename)
        proj_mat = np.zeros(shape=(3, 4, 4), dtype=np.float64)  #
        proj_mat[0, :4, :4] = np.dot(extrinsics,np.linalg.inv(H))
        proj_mat[1, :3, :3] = intrinsics
        proj_mat[2,  0, :4] = depth
        proj_matrices.append(proj_mat)
    return proj_matrices

def write_cam(cam_dir,proj_mat):
        # write
    # try:
    #     os.makedirs(cam_dir)
    # except os.error:
    #     print(cam_dir + ' already exist.')
    for i in range(len(proj_mat)):
        with open(os.path.join(cam_dir, '%08d_cam.txt' % i), 'w') as f:
            extrinsic = proj_mat[i][0, :4, :4]
            f.write('extrinsic\n')
            for j in range(4):
                for k in range(4):
                    f.write(str(extrinsic[j, k]) + ' ')
                f.write('\n')
            f.write('\nintrinsic\n')
            intrinsic = proj_mat[i][1, :3, :3]
            for j in range(3):
                for k in range(3):
                    f.write(str(intrinsic[j, k]) + ' ')
                f.write('\n')
            depth = proj_mat[i][2,  0, :4]
            f.write('\n%f %f %f %f\n' % (depth[0], depth[1], depth[2], depth[3]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert camera to world coord.')

    parser.add_argument('--cams_folder', required=False, default="/media/frankylee/MyPassport/rjwang/AVS3/data/olympic/olympic_train/Camers/cams_olympic/cams/",type=str, help='cams_folder.')
    parser.add_argument('--save_folder', required=False, default="/media/frankylee/MyPassport/rjwang/AVS3/data/olympic/olympic_train/Camers/cams_center_coord/cams/", type=str, help='save_folder.')

    args = parser.parse_args()
    proj_mat = get_cams(args.cams_folder)
    write_cam(args.save_folder,proj_mat)