from IPython.core.debugger import set_trace
from importlib import reload

import numpy as np
import scipy.stats
import cv2.cv2 as cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

import gori_tools as got
reload(got)

class PathMaker(object):
    def __init__(self, mappy, path_params):
        self._mappy = mappy
        self._scale = self._mappy._scale
        self._hall_width = self._mappy._hall_width
        self._safety_buffer = self._hall_width/2
        self._path_memory = path_params['path_memory']
        self._max_dist = path_params['max_traverse_dist']
        self._waypoint_dist_factor = path_params['waypoint_dist_factor']
        self._wall_waypoint_factor = path_params['wall_waypoint_factor']
        self._log_scale_weight = path_params['log_scale_weight']

    @property
    def path_memory(self):
        return self._path_memory

    def smartlyPlaceDots(self):
        stride = int((self._hall_width)//self._scale)
        print("Placing Dots")
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
        waypoint_displacements = np.array([XY_scale[None,:,0] -
                                           XY_scale[:,0,None],
                                           XY_scale[None,:,1] -
                                           XY_scale[:,1,None]])

        waypoint_distances = np.linalg.norm(waypoint_displacements, axis=0)
        # waypoint_angles = np.arctan2(waypoint_displacements[1], waypoint_displacements[0])

        #find waypoints too close to eachother
        idx_bool = waypoint_distances<self._hall_width*self._waypoint_dist_factor
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

        #prune away waypoints that arent near a wall
        min_distances, min_angles = self._mappy.getClosestObstacles(XY_scale)
        idx_bool_far = min_distances > self._safety_buffer*self._wall_waypoint_factor
        idx_far = np.array(np.where(idx_bool_far))
        XY_scale = np.delete(XY_scale,idx_far,0)

        #reset XY for pixel scale
        XY = XY_scale/self._scale
        XY = np.rint(XY).astype(int)
        self._XY = XY

    def computeTraversableGraph(self):
        self._graph = np.zeros((len(self._XY), len(self._XY)))
        displacements = np.array([self._XY[None,:,0] - self._XY[:,0,None],
                                  self._XY[None,:,1] - self._XY[:,1,None]])
        distances = np.linalg.norm(displacements, axis=0)*self._scale
        angles = np.arctan2(displacements[1], displacements[0])
        sector_width = np.pi/4
        sectors = np.array([-7/2,-5/2,-3/2, -1/2, 1/2, 3/2, 5/2, 7/2])
        sectors = sector_width*sectors

        idx_bool = np.zeros((distances.shape[0],distances.shape[1]))
        # idx_bool[:,:] = 0
        # np.zeros_like(distances)
        sectored_angles = np.digitize(angles,sectors)
        sectored_angles[sectored_angles==8]=0
        for ii, distance_row in enumerate(distances):
            traversable_nodes = []
            for sector in range(len(sectors)):
                candidates = np.array(np.where(sectored_angles[ii] == sector)).flatten()
                if sector == 4:
                    candidates = candidates[candidates!=ii]
                if candidates.size > 0:
                    traversable_nodes.append(candidates[np.argmin(distance_row[candidates])])

            idx_bool[ii,traversable_nodes]=True
        idx_bool2 = distances<self._max_dist

        idx_bool_total = np.logical_and(idx_bool,idx_bool2)
        in_range_idx = np.array(np.where(idx_bool_total))
        for ii, edge in enumerate(tqdm(in_range_idx.T, desc="Pruning Collisions")):
            if self._mappy.lineCollisionCheck(self._XY[edge[0]],self._XY[edge[1]],self._safety_buffer/3):
                self._graph[edge[0], edge[1]] = 1

    def saveTraversableGraph(self,file_name):
        np.save(file_name, self._graph, allow_pickle=False, fix_imports=False)

    def loadTraversableGraph(self,file_name):
        self._graph = np.load(file_name)

    def saveWptsXY(self,file_name):
        np.save(file_name, self._XY, allow_pickle=False, fix_imports=False)

    def loadWptsXY(self,file_name):
        self._XY = np.load(file_name)

    def assignWeights(self, current_heading, current_idx, choices):
        displacements = np.array([self._XY[choices,0] - self._XY[current_idx,0],
                                  self._XY[choices,1] - self._XY[current_idx,1]])

        angles = np.arctan2(displacements[1], displacements[0]) - current_heading
        angles = got.rad_wrap_pi(angles)
        w = scipy.stats.norm.logpdf(angles, scale=self._log_scale_weight)
        w -= np.max(w)
        w = np.exp(w)
        w = w / np.sum(w)
        return w

    def makeMeAPath(self, path_len, start_idx, prev_path = None):
        current_idx = start_idx
        current_heading = np.random.rand()*2*np.pi - np.pi
        if prev_path is not None:
            path_idx = prev_path[-self._path_memory:]
        else:
            path_idx = np.array([start_idx])
        cur_path_len = 1
        lop_len = len(path_idx)-1
        for _ in range(path_len):
            choices = (np.where(self._graph[current_idx]))[0]
            choices_comb = np.setdiff1d(choices,path_idx[-self._path_memory:])
            if len(choices_comb) > 0:
                w = self.assignWeights(current_heading, current_idx, choices_comb)
                new_idx = np.random.choice(choices_comb, p=w)
            else:
                w = self.assignWeights(current_heading, current_idx, choices)
                new_idx = np.random.choice(choices, p=w)

            path_idx = np.append(path_idx,new_idx)
            current_idx = new_idx
            cur_path_len += 1

        return path_idx[lop_len:].astype(int)

    def print_graph(self):
        for i in range(self._graph.shape[0]):
            print(i, np.array(np.where(self._graph[i] == 1)))

    @property
    def waypoint_locs(self):
        return self._XY
