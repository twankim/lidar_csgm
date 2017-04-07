"""
Demonstrating how to undistort images.

Reads in the given calibration file, parses it, and uses it to undistort the given
image. Then display both the original and undistorted images.

To use:

    python undistort.py image calibration_file
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
            #chunks = f.readline().rstrip().split(' ')
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
            mapped[0,idx] = np.round(self.mapu[points[0,idx],
                                               points[1,idx]]).astype('int')
            mapped[1,idx] = np.round(self.mapv[points[0,idx],
                                               points[1,idx]]).astype('int')

        idx = (mapped[0,:]>=0) & (mapped[0,:]<self.mapu.shape[0]) &\
              (mapped[1,:]>=0) & (mapped[1,:]<self.mapu.shape[1])
        return mapped[:,idx]
    # def undistort(self, img):
    #     return cv2.resize(cv2.remap(img, self.mapu, self.mapv, cv2.INTER_LINEAR),
    #                       (self.mask.shape[1], self.mask.shape[0]),
    #                       interpolation=cv2.INTER_CUBIC)

def main():
    parser = argparse.ArgumentParser(description="Undistort images")
    parser.add_argument('image', metavar='img', type=str, help='image to undistort')
    parser.add_argument('map', metavar='map', type=str, help='undistortion map')

    args = parser.parse_args()

    undistort = Undistort(args.map)
    print 'Loaded camera calibration'

    im = cv2.imread(args.image)

    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    cv2.imshow('Image', im)

    im_undistorted = undistort.undistort(im)

    cv2.namedWindow('Undistorted Image', cv2.WINDOW_NORMAL)
    cv2.imshow('Undistorted Image', im_undistorted)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
