# !/usr/bin/python
#
# Demonstrates how to project velodyne points to camera imagery. Requires a binary
# velodyne sync file, undistorted image, and assumes that the calibration files are
# in the directory.
#
# To use:
#
#    python project_vel_to_cam.py
# 
#
#       train:  Dates for train
#        test:  Dates for test
#        step:  step size between LIDAR frames
#        init:  Number of frames to skip from the beginning
#        data:  Path of NCLT data
#

import sys
import struct
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
import cv2
from scipy import ndimage
import argparse
import glob
import cPickle

from undistort_lidar import Undistort

# laser ids to be collected (Manually select as 8 layer)
id_lowres = [22,26,30,3,7,11,15,19]

# Size of Image
im_width = 1616
im_height = 1232

def convert(x_s, y_s, z_s):

    scaling = 0.005 # 5 mm
    offset = -100.0

    x = x_s * scaling + offset
    y = y_s * scaling + offset
    z = z_s * scaling + offset

    return x, y, z

def load_vel_hits(filename):

    f_bin = open(filename, "r")

    hits = []
    hits_info = []

    while True:

        x_str = f_bin.read(2)
        if x_str == '': # eof
            break

        x = struct.unpack('<H', x_str)[0]
        y = struct.unpack('<H', f_bin.read(2))[0]
        z = struct.unpack('<H', f_bin.read(2))[0]
        i = struct.unpack('B', f_bin.read(1))[0]
        l = struct.unpack('B', f_bin.read(1))[0]

        x, y, z = convert(x, y, z)

        # Load in homogenous
        hits += [[x, y, z, 1]]
        hits_info += [[i,l]]

    f_bin.close()
    hits = np.asarray(hits)
    hits_info = np.asarray(hits_info)

    return hits.transpose(),hits_info.transpose()

def ssc_to_homo(ssc):

    # Convert 6-DOF ssc coordinate transformation to 4x4 homogeneous matrix
    # transformation

    sr = np.sin(np.pi/180.0 * ssc[3])
    cr = np.cos(np.pi/180.0 * ssc[3])

    sp = np.sin(np.pi/180.0 * ssc[4])
    cp = np.cos(np.pi/180.0 * ssc[4])

    sh = np.sin(np.pi/180.0 * ssc[5])
    ch = np.cos(np.pi/180.0 * ssc[5])

    H = np.zeros((4, 4))

    H[0, 0] = ch*cp
    H[0, 1] = -sh*cr + ch*sp*sr
    H[0, 2] = sh*sr + ch*sp*cr
    H[1, 0] = sh*cp
    H[1, 1] = ch*cr + sh*sp*sr
    H[1, 2] = -ch*sr + sh*sp*cr
    H[2, 0] = -sp
    H[2, 1] = cp*sr
    H[2, 2] = cp*cr

    H[0, 3] = ssc[0]
    H[1, 3] = ssc[1]
    H[2, 3] = ssc[2]

    H[3, 3] = 1

    return H

def project_vel_to_cam(hits, cam_num):

    # Load camera parameters
    K = np.loadtxt('cam_params/K_cam%d.csv' % (cam_num), delimiter=',')
    x_lb3_c = np.loadtxt('cam_params/x_lb3_c%d.csv' % (cam_num), delimiter=',')

    # Other coordinate transforms we need
    x_body_lb3 = [0.035, 0.002, -1.23, -179.93, -0.23, 0.50]

    # Now do the projection
    T_lb3_c = ssc_to_homo(x_lb3_c)
    T_body_lb3 = ssc_to_homo(x_body_lb3)

    T_lb3_body = np.linalg.inv(T_body_lb3)
    T_c_lb3 = np.linalg.inv(T_lb3_c)

    T_c_body = np.matmul(T_c_lb3, T_lb3_body)

    hits_c = np.matmul(T_c_body, hits)
    hits_im = np.matmul(K, hits_c[0:3, :])

    return hits_im

def points_to_image(points,im_shape):
    im_lidar = np.zeros(im_shape,dtype=np.uint8)
    for idx in xrange(points.shape[1]):
        x,y = int(points[0,idx]),int(points[1,idx])
        z_val = 1-min(points[2,idx]/100.0,1.0)
        im_lidar[y,x] = np.round(z_val*255.0).astype('uint8')
    return im_lidar

def gen_data(d_path,list_date,n_step,n_init,name_set):
    set_path = os.path.join(d_path,name_set)
    lr_path = os.path.join(set_path,'layer_8')
    lr_path_im = os.path.join(lr_path,'images')
    lr_path_pt = os.path.join(lr_path,'points')
    hr_path = os.path.join(set_path,'layer_32')
    hr_path_im = os.path.join(hr_path,'images')
    hr_path_pt = os.path.join(hr_path,'points')
    
    # Make required diretories
    if not os.path.exists(set_path):
        os.makedirs(set_path)
    if not os.path.exists(lr_path):
        os.makedirs(lr_path)
    if not os.path.exists(lr_path_im):
        os.makedirs(lr_path_im)
    if not os.path.exists(lr_path_pt):
        os.makedirs(lr_path_pt)
    if not os.path.exists(hr_path):
        os.makedirs(hr_path)
    if not os.path.exists(hr_path_im):
        os.makedirs(hr_path_im)
    if not os.path.exists(hr_path_pt):
        os.makedirs(hr_path_pt)

    for d_set in list_date:
        curr_path = os.path.join(d_path,d_set)
        velo_path = os.path.join(curr_path,'velodyne_sync/')
        
        # Load velodyne files
        f_lidar = glob.glob(os.path.join(velo_path,'*.bin'))
        name_list = [lname.split(velo_path)[1].split('.')[0] for lname in f_lidar]
        
        # Skip n_init_points and 
        assert n_init<len(name_list),\
               'n_init must be less than the number of files: {}'.format(curr_path)
        name_list = name_list[n_init::n_step]

        for fname in name_list:
            # Load velodyne points
            hits_body,hits_info = load_vel_hits(os.path.join(velo_path,fname+'.bin'))

            for cam_num in xrange(1,6):
                hits_image = project_vel_to_cam(hits_body,cam_num)

                # x,y,z,i,l
                points = np.vstack((hits_image,hits_info))
                points[0,:] = np.round(hits_image[0, :]/hits_image[2, :])
                points[1,:] = np.round(hits_image[1, :]/hits_image[2, :])

                idx_incam = (points[0,:]>=0) & (points[0,:]<im_width) &\
                            (points[1,:]>=0) & (points[1,:]<im_height) &\
                            (points[2,:]>0)
                idx_lowres = [points[4,i] in id_lowres for i in xrange(points.shape[1])]\
                             & idx_incam

                points_lr = points[:,idx_lowres] # Low resolution (8layser)
                points = points[:,idx_incam]

                undistort = Undistort(os.path.join(
                                        d_path,
                                        'cam_params/U2D_Cam{}_1616X1232.txt'.format(cam_num)
                                     ))
                points_ud = undistort.undistort(points)
                points_lr_ud = undistort.undistort(points_lr)

                im_lidar_ud = points_to_image(points_ud,(im_height,im_width))
                im_lidar_lr_ud = points_to_image(points_lr_ud,(im_height,im_width))

                # # Load image
                # image = cv2.imread(os.path.join(
                #                         d_path,
                #                         'lb3',
                #                         'Cam{}'.format(cam_num),
                #                         fname+'.tiff'
                #                    ))[:,:,(2,1,0)]

                # plot_imlidar(image,points_ud,points_lr_ud)


def gen_lidar_patch():
    with open(temp_file,'wb') as fid:
        cPickle.dump([aa,bb], fid, cPickle.HIGHEST_PROTOCOL)
    
    cv2.imwrite('lidar_undist_cam{}.png'.format(cam_num),im_lidar_ud)
    cv2.imwrite('lidar_undist_cam{}_lr.png'.format(cam_num),im_lidar_lr_ud)

def plot_imlidar(image,points,points_lr):
    # Plot with image
    plt.figure(1)
    plt.subplot(1,2,1)
    plt.imshow(ndimage.rotate(image,270))
    plt.hold(True)
    plt.scatter(im_height-points[1,:]-1,
                points[0,:],
                c=points[2,:],
                s=5,
                linewidths=0)
    plt.xlim(0, im_height)
    plt.ylim(im_width, 0)
    plt.title('32-layer Laser')

    plt.subplot(1,2,2)
    plt.imshow(ndimage.rotate(image,270))
    plt.hold(True)
    plt.scatter(im_height-points_lr[1,:]-1,
                points_lr[0,:],
                c=points_lr[2,:],
                s=5,
                linewidths=0)
    plt.xlim(0, im_height)
    plt.ylim(im_width, 0)
    plt.title('8-layer Laser')

    plt.show()

def parse_args():
    parser = argparse.ArgumentParser(description=
                        'Generate LIDAR patch dataset for GAN')
    parser.add_argument('-train', dest='f_train',
                        help='Dates for train',
                        default = '2012-04-29', type = str)
    parser.add_argument('-test', dest='f_test',
                        help='Dates for test',
                        default = '2013-01-10', type = str)
    parser.add_argument('-step', dest='step',
                        help='Step size between LIDAR frames',
                        default = 2, type = int)
    parser.add_argument('-init', dest='n_init',
                        help='Number of frames to skip',
                        default = 100, type = int)
    parser.add_argument('-data', dest='d_path',
                        help='Path of NCLT data',
                        default = os.getcwd(), type = str)
    # parser.add_argument('-cam', dest='cam_num',
    #                     help='Index of the camera (1~5)',
    #                     default = 5, type = int)
    args = parser.parse_args()
    return args

def main(args):
    list_train = args.f_train.split(',')
    list_test = args.f_test.split(',')
    n_step = args.step
    n_init = args.n_init
    if args.d_path[0] == '/':
        d_path = args.d_path # Absolute path
    else:
        d_path = os.path.join(os.getcwd(),args.d_path)
    
    # ----------- Train Data ----------------
    gen_data(d_path,list_train,n_step,n_init,'train')

    # ----------- Test Data ----------------

    return 0

if __name__ == '__main__':
    args = parse_args()
    print "Called with args:"
    print args
    sys.exit(main(args))
