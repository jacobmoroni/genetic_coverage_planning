from IPython.core.debugger import set_trace
from importlib import reload

import numpy as np
import scipy.stats
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

import gori_tools as got
reload(got)

class PathMaker(object):
    def __init__(self, mappy, scale, hall_width, safety_buffer):
        self._mappy = mappy
        self._scale = scale
        self._hall_width = hall_width
        self._safety_buffer = hall_width/2#safety_buffer
        size = self._mappy.shape
        self._path_memory = 10
        # self._grid = np.mgrid[0:size[0]*scale:scale, 0:size[1]*scale:scale]
    #
    @property
    def path_memory(self):
        return self._path_memory

    def smartlyPlaceDots(self):
        self.pac_dots = np.zeros_like(self._mappy)
        stride = int((self._hall_width)//self._scale)
        print("Placing Dots", self.pac_dots.shape)
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
        print("Pruning Dots")
        waypoint_displacements = np.array([XY_scale[None,:,0] - XY_scale[:,0,None],
                                  XY_scale[None,:,1] - XY_scale[:,1,None]])
        #
        waypoint_distances = np.linalg.norm(waypoint_displacements, axis=0)
        # waypoint_angles = np.arctan2(waypoint_displacements[1], waypoint_displacements[0])

        #find waypoints too close to eachother
        idx_bool = waypoint_distances<self._hall_width*0.8#0.9#0.7
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

    def computeTraversableGraph(self, max_dist):
        self._graph = np.zeros((len(self._XY), len(self._XY)))
        displacements = np.array([self._XY[None,:,0] - self._XY[:,0,None],
                                  self._XY[None,:,1] - self._XY[:,1,None]])
        distances = np.linalg.norm(displacements, axis=0)*self._scale
        angles = np.arctan2(displacements[1], displacements[0])
        sector_width = np.pi/4
        idx_bool = np.zeros_like(distances)
        for ii, (angle_row, distance_row) in enumerate(zip(angles,distances)):
            sector1 = np.array(np.where(np.logical_and(angle_row<(sector_width* 1/2), angle_row>(sector_width*-1/2)))).flatten()
            sector2 = np.array(np.where(np.logical_and(angle_row<(sector_width* 3/2), angle_row>(sector_width* 1/2)))).flatten()
            sector3 = np.array(np.where(np.logical_and(angle_row<(sector_width* 5/2), angle_row>(sector_width* 3/2)))).flatten()
            sector4 = np.array(np.where(np.logical_and(angle_row<(sector_width* 7/2), angle_row>(sector_width* 5/2)))).flatten()
            sector5 = np.array(np.where(np.logical_and(angle_row>(sector_width*-3/2), angle_row<(sector_width*-1/2)))).flatten()
            sector6 = np.array(np.where(np.logical_and(angle_row>(sector_width*-5/2), angle_row<(sector_width*-3/2)))).flatten()
            sector7 = np.array(np.where(np.logical_and(angle_row>(sector_width*-7/2), angle_row<(sector_width*-5/2)))).flatten()
            sector8 = np.array(np.where(np.logical_or(angle_row>(sector_width* 7/2), angle_row<(sector_width*-7/2)))).flatten()
            sector1 = np.delete(sector1,np.where(sector1==ii))
            traversable_nodes = []
            if sector1.size > 0:
                traversable_nodes.append(sector1[np.argmin(distance_row[sector1])])
            if sector2.size > 0:
                traversable_nodes.append(sector2[np.argmin(distance_row[sector2])])
            if sector3.size > 0:
                traversable_nodes.append(sector3[np.argmin(distance_row[sector3])])
            if sector4.size > 0:
                traversable_nodes.append(sector4[np.argmin(distance_row[sector4])])
            if sector5.size > 0:
                traversable_nodes.append(sector5[np.argmin(distance_row[sector5])])
            if sector6.size > 0:
                traversable_nodes.append(sector6[np.argmin(distance_row[sector6])])
            if sector7.size > 0:
                traversable_nodes.append(sector7[np.argmin(distance_row[sector7])])
            if sector8.size > 0:
                traversable_nodes.append(sector8[np.argmin(distance_row[sector8])])
            idx_bool[ii,traversable_nodes]=True

        # set_trace()
        idx_bool2 = distances<max_dist
        idx_bool_total = np.logical_and(idx_bool,idx_bool2)
        # print(f"idx_bool: {idx_bool.shape}")
        in_range_idx = np.array(np.where(idx_bool_total))
        # print(f"in_range_idx: {in_range_idx.shape}")
        for ii, edge in enumerate(tqdm(in_range_idx.T, desc="Pruning Collisions")):
            if self._mappy.lineCollisionCheck(self._XY[edge[0]],self._XY[edge[1]],self._safety_buffer/3):
                self._graph[edge[0], edge[1]] = 1
            #
        #
        # self._graph[in_range_idx[0], in_range_idx[1]] = 1
    #
    def saveTraversableGraph(self,file_name):
        np.save(file_name, self._graph, allow_pickle=False, fix_imports=False)
    #
    def loadTraversableGraph(self,file_name):
        self._graph = np.load(file_name)
    #
    def saveWptsXY(self,file_name):
        np.save(file_name, self._XY, allow_pickle=False, fix_imports=False)
    #
    def loadWptsXY(self,file_name):
        self._XY = np.load(file_name)
    #
    def assignWeights(self, current_heading, current_idx, choices):
        displacements = np.array([self._XY[choices,0] - self._XY[current_idx,0],
                                  self._XY[choices,1] - self._XY[current_idx,1]])
        #
        angles = np.arctan2(displacements[1], displacements[0]) - current_heading
        angles = got.rad_wrap_pi(angles)
        w = scipy.stats.norm.logpdf(angles, scale=1)
        w -= np.max(w)
        w = np.exp(w)
        w = w / np.sum(w)
        return w
    #
    def makeMeAPath(self,path_length,start_idx):
        current_idx = start_idx
        current_heading = np.random.rand()*2*np.pi - np.pi;
        path_idx = np.array([start_idx])
        while len(path_idx)<path_length:
            choices = (np.where(self._graph[current_idx]))[0]
            choices_comb = np.setdiff1d(choices,path_idx[-self._path_memory:])
            if len(choices_comb) > 0:
                w = self.assignWeights(current_heading, current_idx, choices_comb)
                new_idx = np.random.choice(choices_comb, p=w)
                current_heading = np.arctan2(self._XY[new_idx,1] - self._XY[current_idx,1],
                                             self._XY[new_idx,0] - self._XY[current_idx,0])
            else:
                # set_trace()
                w = self.assignWeights(current_heading, current_idx, choices)
                new_idx = np.random.choice(choices, p=w)
                current_heading = np.arctan2(self._XY[new_idx,1] - self._XY[current_idx,1],
                                             self._XY[new_idx,0] - self._XY[current_idx,0])
            #
            path_idx = np.append(path_idx,new_idx)
            current_idx = new_idx
            # print (current_idx)
        #
        return path_idx.astype(int)

    @property
    def waypoint_locs(self):
        return self._XY
    #
#
