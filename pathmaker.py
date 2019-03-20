from IPython.core.debugger import set_trace
from importlib import reload

import numpy as np
import cv2
import matplotlib.pyplot as plt

class PathMaker(object):
    def __init__(self, mappy, scale, hall_width, safety_buffer):
        self._mappy = mappy/255
        self._scale = scale
        self._hall_width = hall_width
        self._safety_buffer = safety_buffer
        size = self._mappy.shape
        # self._grid = np.mgrid[0:size[0]*scale:scale, 0:size[1]*scale:scale]
    #
    def smartly_place_dots(self):
        self.pac_dots = np.zeros_like(self._mappy)
        stride = int((self._hall_width)//self._scale)
        print(self.pac_dots.shape)
        size = self._mappy.shape
        X,Y = np.mgrid[0:size[0]:stride,0:size[1]:stride]
        XY = np.vstack((X.flatten(), Y.flatten())).T
        XY = XY[self._mappy[XY[:,0], XY[:,1]] == 0]
        XY_scale = XY*self._scale
        # now we need to move dots to be at least safety_buffer away from obstacles
        obstacles = (np.array(np.nonzero(self._mappy)) * self._scale)
        # distances = np.zeros((XY.shape[0], obstacles.shape[0]))
        # angles = np.zeros_like(distances)
        displacements = np.array([obstacles[None, 0] - XY_scale[:,0, None], obstacles[None, 1] - XY_scale[:,1, None]])
        print(displacements.shape)
        distances = np.linalg.norm(displacements, axis=0)
        angles = np.arctan2(displacements[1], displacements[0])
        print(distances.shape)
        print(angles.shape)
        # for i, obs in enumerate(XY):
        #
        #     for j, point in enumerate(obstacles):
        #         distances[i,j] = np.linalg.norm(obs - point);
        #         angles[i,j] = np.arctan2(obs[1] - point[1], obs[0] - point[0])

        closest = np.argmin(distances, axis=1)
        print(closest.shape)
        min_distances = distances[range(len(closest)),closest]
        min_angles = angles[range(len(closest)),closest]
        idx_bool = min_distances<self._safety_buffer
        idx = np.array(np.where(idx_bool))
        # set_trace()
        XY_scale[idx,0] = XY_scale[idx,0] - (self._safety_buffer-min_distances[idx])*np.cos(min_angles[idx])
        XY_scale[idx,1] = XY_scale[idx,1] - (self._safety_buffer-min_distances[idx])*np.sin(min_angles[idx])
        # set_trace()
        XY = XY_scale*1/self._scale
        XY = XY.astype(int)
        num_dilations = int(self._safety_buffer/self._scale)
        kernel = np.ones((3,3),np.uint8)
        map_dilated = cv2.dilate(self._mappy,kernel,iterations = num_dilations)
        self.pac_dots[XY[:,0], XY[:,1]] = 1
        # self.pac_dots = self.pac_dots*(1-self._mappy)
        cv2.imshow('pac_dots',self.pac_dots + 0.25*self._mappy + 0.25*map_dilated)
        cv2.waitKey()
    #
#
