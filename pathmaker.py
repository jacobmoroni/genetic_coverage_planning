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
        self._safety_buffer = hall_width/2#safety_buffer
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
        # print(displacements.shape)
        distances = np.linalg.norm(displacements, axis=0)
        angles = np.arctan2(displacements[1], displacements[0])
        # print(distances.shape)
        # print(angles.shape)



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
        XY = np.rint(XY).astype(int)
        self._XY = XY

    def compute_traversable_graph(self, max_dist):
        self._graph = np.zeros((len(self._XY), len(self._XY)))
        displacements = np.array([self._XY[None,:,0] - self._XY[:,0,None], self._XY[None,:,1] - self._XY[:,1,None]])
        distances = np.linalg.norm(displacements, axis=0)*self._scale
        angles = np.arctan2(displacements[1], displacements[0])

        idx_bool = distances<max_dist
        # print(f"idx_bool: {idx_bool.shape}")
        in_range_idx = np.array(np.where(idx_bool))
        print(f"in_range_idx: {in_range_idx.shape}")
        for i, edge in enumerate(in_range_idx.T):
            # check to see if path is clear
            self._graph[edge[0], edge[1]] = 1
        # self._graph[in_range_idx[0], in_range_idx[1]] = 1

    def visualize_waypoints(self):
        # do some visualization
        num_dilations = int(self._safety_buffer/self._scale)
        kernel = np.ones((3,3),np.uint8)
        map_dilated = cv2.dilate(self._mappy,kernel,iterations = num_dilations)
        self.pac_dots[self._XY[:,0], self._XY[:,1]] = 1
        # self.pac_dots = self.pac_dots*(1-self._mappy)
        img = self.pac_dots + 0.25*self._mappy + 0.25*map_dilated
        img_color = img[...,None]*np.array([1, 1, 1])
        start_idx = 246
        # draw the starting point
        cv2.circle(img_color, (self._XY[start_idx,1], self._XY[start_idx,0]), 5, (0,0,1))
        # img_color[self._XY[start_idx,0], self._XY[start_idx,1], :] = np.array([0,0,1])
        cv2.imshow('pac_dots',img_color)
        cv2.waitKey()
    #
#
