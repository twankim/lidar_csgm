"""
Demonstrating how to undistort images.

Reads in the given calibration file, parses it, and uses it to 
undistort the given lidar point clouds
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse
import re

class Undistort(object):
    def __init__(self, fin):
        self.fin = fin
        # read in distort
        with open(fin, 'r') as f:
            header = f.readline().rstrip()
            chunks = re.sub(r'[^0-9,]', '', header).split(',')
            self.mapu = np.zeros((int(chunks[1]),int(chunks[0])),
                    dtype=np.float32)
            self.mapv = np.zeros((int(chunks[1]),int(chunks[0])),
                    dtype=np.float32)
            for line in f.readlines():
                chunks = line.rstrip().split(' ')
                self.mapu[int(chunks[0]),int(chunks[1])] = float(chunks[3])
                self.mapv[int(chunks[0]),int(chunks[1])] = float(chunks[2])

    """
    Use OpenCV to undistorted the given Lidar Data in Camera space
    """
    def undistort(self, points):
        mapped = np.zeros(np.shape(points))
        mapped[2:,:] = points[2:,:]
        for idx in xrange(np.shape(points)[1]):
            mapped[0,idx] = np.round(self.mapu[int(points[1,idx]),
                                               int(points[0,idx])])\
                                    .astype('int')
            mapped[1,idx] = np.round(self.mapv[int(points[1,idx]),
                                               int(points[0,idx])])\
                                    .astype('int')

        idx = (mapped[0,:]>=0) & (mapped[0,:]<self.mapu.shape[1]) &\
              (mapped[1,:]>=0) & (mapped[1,:]<self.mapu.shape[0])
        return mapped[:,idx]
