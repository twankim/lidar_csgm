# !/usr/bin/python
#
# Demonstrates how to project velodyne points to camera imagery. Requires a binary
# velodyne sync file, undistorted image, and assumes that the calibration files are
# in the directory.
#
# To use:
#
#    python project_vel_to_cam.py vel img cam_num
#
#       vel:  The velodyne binary file (timestamp.bin)
#       img:  The undistorted image (timestamp.tiff)
#   cam_num:  The index (0 through 5) of the camera
#    d_path:  The path for nclt data
#

import sys
import struct
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
import cv2
from scipy import ndimage

from undistort_lidar import Undistort

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

def plot_imlidar(image,points,points_lr):
    # Plot with image
    plt.figure(1)
    plt.subplot(1,2,1)
    plt.imshow(ndimage.rotate(image,270))
    plt.hold(True)
    plt.scatter(1231-points[1,:],
                points[0,:],
                c=points[2,:],
                s=5,
                linewidths=0)
    plt.xlim(0, 1232)
    plt.ylim(1616, 0)
    plt.title('32-layer Laser')

    plt.subplot(1,2,2)
    plt.imshow(ndimage.rotate(image,270))
    plt.hold(True)
    plt.scatter(1231-points_lr[1,:],
                points_lr[0,:],
                c=points_lr[2,:],
                s=5,
                linewidths=0)
    plt.xlim(0, 1232)
    plt.ylim(1616, 0)
    plt.title('8-layer Laser')

    plt.show()

def main(args):

    if len(args)<4:
        print  """Incorrect usage.

To use:

   python project_vel_to_cam.py vel img cam_num d_path

      vel:  The velodyne binary file (timestamp.bin)
      img:  The undistorted image (timestamp.tiff)
  cam_num:  The index (0 through 5) of the camera
   d_path:  The manual path of the data
"""
        return 1

    if len(args)>4:
        d_path = args[4] # Use specified path
    else:
        d_path = os.getcwd() # Use current path

    # Load velodyne points
    hits_body,hits_info = load_vel_hits(os.path.join(d_path,args[1]))

    # Load image
    image = cv2.imread(os.path.join(d_path,args[2]))[:,:,(2,1,0)]

    cam_num = int(args[3])

    hits_image = project_vel_to_cam(hits_body, cam_num)
    
    # laser ids to be collected (Manually select as 8 layer)
    id_lowres = [22,26,30,3,7,11,15,19]

    # x,y,z,i,l
    points = np.vstack((hits_image,hits_info))
    points[0,:] = np.round(hits_image[0, :]/hits_image[2, :])
    points[1,:] = np.round(hits_image[1, :]/hits_image[2, :])

    idx_incam = (points[0,:]>=0) & (points[0,:]<image.shape[1]) &\
                (points[1,:]>=0) & (points[1,:]<image.shape[0]) &\
                (points[2,:]>0)
    idx_lowres = [points[4,i] in id_lowres for i in xrange(points.shape[1])]\
                 & idx_incam

    points_lr = points[:,idx_lowres] # Low resolution (8layser)
    points = points[:,idx_incam]

    # im_lidar = points_to_image(points,np.shape(image)[0:2])
    # im_lidar_lr = points_to_image(points_lr,np.shape(image)[0:2])

    # cv2.imwrite('lidar_dist_cam{}.png'.format(cam_num),im_lidar)
    # cv2.imwrite('lidar_dist_cam{}_lr.png'.format(cam_num),im_lidar_lr)

    undistort = Undistort('cam_params/U2D_Cam{}_1616X1232.txt'.format(cam_num))
    points_ud = undistort.undistort(points)
    points_lr_ud = undistort.undistort(points_lr)

    im_lidar_ud = points_to_image(points_ud,np.shape(image)[0:2])
    im_lidar_lr_ud = points_to_image(points_lr_ud,np.shape(image)[0:2])
    
    cv2.imwrite('lidar_undist_cam{}.png'.format(cam_num),im_lidar_ud)
    cv2.imwrite('lidar_undist_cam{}_lr.png'.format(cam_num),im_lidar_lr_ud)

    plot_imlidar(image,points_ud,points_lr_ud)

    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv))
