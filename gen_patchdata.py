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

# Minimum number of points in a patch (c_ratio*p_size)
c_ratio = 0.5

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

def project_vel_to_cam(hits,d_path,cam_num):

    # Load camera parameters
    K = np.loadtxt(os.path.join(d_path,'cam_params/K_cam%d.csv' % (cam_num)), delimiter=',')
    x_lb3_c = np.loadtxt(os.path.join(d_path,'cam_params/x_lb3_c%d.csv' % (cam_num)), delimiter=',')

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

def gen_data(d_path,list_date,n_step,n_init,name_set,p_size,cam_nums):
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
        n_total = len(name_list)

        for i_iter,fname in enumerate(name_list):
            print "    ... Generating: {} ({}/{})".format(name_set,i_iter+1,n_total)
            # Load velodyne points
            hits_body,hits_info = load_vel_hits(os.path.join(velo_path,fname+'.bin'))

            for cam_num in cam_nums:
                hits_image = project_vel_to_cam(hits_body,d_path,cam_num)

                # x,y,z,i,l
                points = np.vstack((hits_image,hits_info))
                points[0,:] = np.round(hits_image[0, :]/hits_image[2, :])
                points[1,:] = np.round(hits_image[1, :]/hits_image[2, :])

                idx_incam = (points[0,:]>=0) & (points[0,:]<im_width) &\
                            (points[1,:]>=0) & (points[1,:]<im_height) &\
                            (points[2,:]>0)
                idx_lowres = [points[4,i] in id_lowres for i in xrange(points.shape[1])]\
                             & idx_incam

                points_lr = points[:,idx_lowres] # Low resolution (8-layer)
                points = points[:,idx_incam] # High resolution (32-layer)

                # Undistort images
                undistort = Undistort(os.path.join(
                                        d_path,
                                        'cam_params/U2D_Cam{}_1616X1232.txt'.format(cam_num)
                                     ))
                points_ud = undistort.undistort(points) # High resolution (32-layer)
                points_lr_ud = undistort.undistort(points_lr) # Low resolution (8-layer)

                im_lidar = points_to_image(points_ud,(im_height,im_width))
                im_lidar_lr = points_to_image(points_lr_ud,(im_height,im_width))

                if name_set == 'train':
                    list_pidx,list_pidx_lr,list_pinit = get_patch_idx(
                                                            points_ud,
                                                            points_lr_ud,
                                                            p_size)
                    for i in xrange(len(list_pidx)):
                        # Save Image patches
                        p_name = '{}_{}_{}'.format(fname,cam_num,i)
                        im_path = os.path.join(hr_path_im,p_name+'.png')
                        im_path_lr = os.path.join(lr_path_im,p_name+'.png')
                                                                    
                        x_init,y_init = list_pinit[i]
                        cv2.imwrite(im_path,
                                    im_lidar[y_init:y_init+p_size,x_init:x_init+p_size])
                        cv2.imwrite(im_path_lr,
                                    im_lidar_lr[y_init:y_init+p_size,x_init:x_init+p_size])

                        # Save Points
                        pt_path = os.path.join(hr_path_pt,p_name+'.pkl')
                        pt_path_lr = os.path.join(lr_path_pt,p_name+'.pkl')
                        with open(pt_path,'wb') as fid:
                            cPickle.dump(points_ud[:,list_pidx[i]], fid, cPickle.HIGHEST_PROTOCOL)
                        with open(pt_path_lr,'wb') as fid:
                            cPickle.dump(points_lr_ud[:,list_pidx_lr[i]], fid, cPickle.HIGHEST_PROTOCOL)
                else:
                    # For test set, save original LIDAR image (32/8) and corresponding points
                    p_name = '{}_{}'.format(fname,cam_num,i)
                    im_path = os.path.join(hr_path_im,p_name+'.png')
                    im_path_lr = os.path.join(lr_path_im,p_name+'.png')
                    cv2.imwrite(im_path,im_lidar)
                    cv2.imwrite(im_path_lr,im_lidar_lr)

                    pt_path = os.path.join(hr_path_pt,p_name+'.pkl')
                    pt_path_lr = os.path.join(lr_path_pt,p_name+'.pkl')
                    with open(pt_path,'wb') as fid:
                        cPickle.dump(points_ud, fid, cPickle.HIGHEST_PROTOCOL)
                    with open(pt_path_lr,'wb') as fid:
                        cPickle.dump(points_lr_ud, fid, cPickle.HIGHEST_PROTOCOL)

                # # Load image
                # image = cv2.imread(os.path.join(
                #                         d_path,
                #                         'lb3',
                #                         'Cam{}'.format(cam_num),
                #                         fname+'.tiff'
                #                    ))[:,:,(2,1,0)]

                # plot_imlidar(image,points_ud,points_lr_ud)

def get_patch_idx(points,points_lr,p_size):
    # x_min,y_min,x_max,y_max = points[0,:].min(),points[1,:].min(),\
    #                           points[0,:].max(),points[1,:].max()
    # x_min_lr,y_min_lr,x_max_lr,y_max_lr = points_lr[0,:].min(),points_lr[1,:].min(),\
    #                                       points_lr[0,:].max(),points_lr[1,:].max()
    list_pidx = []
    list_pidx_lr = []
    list_pinit = []
    for y_init in xrange(0,im_height-p_size,p_size):
        for x_init in xrange(0,im_width-p_size,p_size):
            idx_lr = (points_lr[1,:]>=y_init) & (points_lr[1,:]<y_init+p_size) & \
                     (points_lr[0,:]>=x_init) & (points_lr[0,:]<x_init+p_size)
            # if sum(idx_lr)> p_size*c_ratio:
            # # Consider only patches with enough points after compression
            #     list_pidx_lr.append(idx_lr)
            #     idx = (points[1,:]>=y_init) & (points[1,:]<y_init+p_size) & \
            #           (points[0,:]>=x_init) & (points[0,:]<x_init+p_size)
            #     list_pidx.append(idx)
            #     list_pinit.append([x_init,y_init])
            
            idx = (points[1,:]>=y_init) & (points[1,:]<y_init+p_size) & \
                  (points[0,:]>=x_init) & (points[0,:]<x_init+p_size)
            if sum(idx) >= p_size*c_ratio:
                list_pidx_lr.append(idx_lr)
                list_pidx.append(idx)
                list_pinit.append([x_init,y_init])

    return list_pidx,list_pidx_lr,list_pinit

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
                        help='Dates for train ex) 2012-04-29,2013-01-10',
                        default = '2012-04-29', type = str)
    parser.add_argument('-test', dest='f_test',
                        help='Dates for test ex) 2012-04-29,2013-01-10',
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
    parser.add_argument('-size', dest='p_size',
                        help='Size of a patch p x p',
                        default = 128, type = int)
    parser.add_argument('-cam', dest='cam_nums',
                        help='Index of the camera (1~5) ex) 1,3,5',
                        default = '1,4,5', type = str)
    args = parser.parse_args()
    return args

def main(args):
    list_train = args.f_train.split(',')
    list_test = args.f_test.split(',')
    n_step = args.step
    n_init = args.n_init
    cam_nums = [int(cam_num) for cam_num in args.cam_nums.split(',')]
    if args.d_path[0] == '/':
        d_path = args.d_path # Absolute path
    else:
        d_path = os.path.join(os.getcwd(),args.d_path)
    p_size = args.p_size
    
    # ----------- Train Data ----------------
    print "... Generating Train Data"
    gen_data(d_path,list_train,n_step,n_init,'train',p_size,cam_nums)

    # ----------- Test Data ----------------
    print "... Generating Test Data"
    gen_data(d_path,list_train,n_step,n_init,'test',p_size,cam_nums)

    return 0

if __name__ == '__main__':
    args = parse_args()
    print "Called with args:"
    print args
    sys.exit(main(args))
