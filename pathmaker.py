from IPython.core.debugger import set_trace
from importlib import reload

import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

class PathMaker(object):
    def __init__(self, mappy, scale, hall_width, safety_buffer):
        self._mappy = mappy
        self._scale = scale
        self._hall_width = hall_width
        self._safety_buffer = hall_width/2#safety_buffer
        size = self._mappy.shape
        self._path_memory = 5
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
        # for row in angles:
        sector1 = np.array(np.where(np.logical_and(angles<(sector_width* 1/2), angles>(sector_width*-1/2))))
        sector2 = np.array(np.where(np.logical_and(angles<(sector_width* 3/2), angles>(sector_width* 1/2))))
        sector3 = np.array(np.where(np.logical_and(angles<(sector_width* 5/2), angles>(sector_width* 3/2))))
        sector4 = np.array(np.where(np.logical_and(angles<(sector_width* 7/2), angles>(sector_width* 5/2))))
        sector5 = np.array(np.where(np.logical_and(angles>(sector_width*-3/2), angles<(sector_width*-1/2))))
        sector6 = np.array(np.where(np.logical_and(angles>(sector_width*-5/2), angles<(sector_width*-3/2))))
        sector7 = np.array(np.where(np.logical_and(angles>(sector_width*-7/2), angles<(sector_width*-5/2))))
        sector8 = np.array(np.where(np.logical_and(angles>(sector_width* 7/2), angles<(sector_width*-7/2))))

        # set_trace()
        distance1 = np.argmin(distances[sector1], axis = 1)
        distance2 = np.argmin(distances[sector2], axis = 1)
        distance3 = np.argmin(distances[sector3], axis = 1)
        distance4 = np.argmin(distances[sector4], axis = 1)
        distance5 = np.argmin(distances[sector5], axis = 1)
        distance6 = np.argmin(distances[sector6], axis = 1)
        distance7 = np.argmin(distances[sector7], axis = 1)
        distance8 = np.argmin(distances[sector8], axis = 1)

        # set_trace()
        idx_bool = distances<max_dist
        # print(f"idx_bool: {idx_bool.shape}")
        in_range_idx = np.array(np.where(idx_bool))
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
    def makeMeAPath(self,path_length,start_idx):
        current_idx = start_idx
        path_idx = np.array([start_idx])
        while len(path_idx)<path_length:
            choices = (np.where(self._graph[current_idx]))[0]
            choices_comb = np.setdiff1d(choices,path_idx[-self._path_memory:])
            if len(choices_comb) > 0:
                new_idx = np.random.choice(choices_comb)
            else:
                # set_trace()
                new_idx = np.random.choice(choices)
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
