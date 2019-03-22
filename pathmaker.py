from IPython.core.debugger import set_trace
from importlib import reload

import numpy as np
import cv2
import matplotlib.pyplot as plt

class PathMaker(object):
    def __init__(self, mappy, scale, hall_width, safety_buffer):
        self._mappy = mappy
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
        XY = XY[self._mappy._img[XY[:,0], XY[:,1]] == 0]
        XY_scale = XY*self._scale
        min_distances, min_angles = self._mappy.getClosestObstacles(XY_scale)
        idx_bool = min_distances<self._safety_buffer
        idx = np.array(np.where(idx_bool))
        XY_scale[idx,0] = XY_scale[idx,0] - (self._safety_buffer-min_distances[idx])*np.cos(min_angles[idx])
        XY_scale[idx,1] = XY_scale[idx,1] - (self._safety_buffer-min_distances[idx])*np.sin(min_angles[idx])

        #pruning hallways
        waypoint_displacements = np.array([XY_scale[None,:,0] - XY_scale[:,0,None],
                                  XY_scale[None,:,1] - XY_scale[:,1,None]])

        waypoint_distances = np.linalg.norm(waypoint_displacements, axis=0)
        waypoint_angles = np.arctan2(waypoint_displacements[1], waypoint_displacements[0])

        #find waypoints too close to eachother
        idx_bool = waypoint_distances<self._hall_width*0.9#0.7
        waypoint_I = np.eye(XY_scale.shape[0])
        idx_bool = np.logical_xor(waypoint_I,idx_bool)
        pruning_idx = np.array(np.where(idx_bool)).T
        pruning_idx = np.unique(np.sort(pruning_idx),axis =0)
        #prune waypoints where 2 are close to a single waypoint
        unq, _, unq_count = np.unique(np.sort(pruning_idx,axis = None), return_inverse=True, return_counts=True)
        count_mask = unq_count > 1
        dup_ids = unq[count_mask]
        for num in dup_ids:
            row, _ = np.where(pruning_idx == num)
            prune = pruning_idx[row][pruning_idx[row] != num]
            XY_scale[prune] = -1
            pruning_idx[row] = -1
        row,_ = np.where(pruning_idx == -1)
        pruning_idx = np.delete(pruning_idx,row,0)

        #where 2 are close to eachother, move one halfway and delete the other
        for wp in pruning_idx:
            XY_scale[wp[0],0] = XY_scale[wp[0],0] - (XY_scale[wp[0],0]- XY_scale[wp[1],0])/2
            XY_scale[wp[0],1] = XY_scale[wp[0],1] - (XY_scale[wp[0],1]- XY_scale[wp[1],1])/2
            XY_scale[wp[1]] = -1

        row,_ = np.where(XY_scale == -1)
        XY_scale = np.delete(XY_scale,row,0)

        XY = XY_scale/self._scale
        XY = np.rint(XY).astype(int)
        self._XY = XY

    def compute_traversable_graph(self, max_dist):
        self._graph = np.zeros((len(self._XY), len(self._XY)))
        displacements = np.array([self._XY[None,:,0] - self._XY[:,0,None],
                                  self._XY[None,:,1] - self._XY[:,1,None]])
        distances = np.linalg.norm(displacements, axis=0)*self._scale
        angles = np.arctan2(displacements[1], displacements[0])

        idx_bool = distances<max_dist
        # print(f"idx_bool: {idx_bool.shape}")
        in_range_idx = np.array(np.where(idx_bool))
        # print(f"in_range_idx: {in_range_idx.shape}")
        for i, edge in enumerate(in_range_idx.T):
            if self._mappy.lineCollisionCheck(self._XY[edge[0]],self._XY[edge[1]],self._safety_buffer/2):
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
        start_idx = 201
        check_idx = 359
        # draw the starting point
        cv2.circle(img_color, (self._XY[start_idx,1], self._XY[start_idx,0]), 5, (0,0,1))
        # cv2.circle(img_color, (self._XY[check_idx,1], self._XY[check_idx,0]), 5, (1,0,1))
        # img_color[self._XY[start_idx,0], self._XY[start_idx,1], :] = np.array([0,0,1])
        cv2.imshow('pac_dots',img_color)
        cv2.waitKey()
    #
#
