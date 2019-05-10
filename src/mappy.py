from IPython.core.debugger import set_trace
from importlib import reload

import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import gridspec

import gori_tools as got
reload(got)

class Mappy(object):
    def __init__(self, img, scale, hall_width, min_view, max_view, view_angle, rho):
        self._img = img
        self._num_occluded = np.sum(self._img)
        self._scale = scale
        self._hall_width = hall_width
        self._safety_buffer = hall_width/2
        self.shape = self._img.shape

        # view area stuff for coverage calcualtion
        self._num_rays = 15
        self._min_view = min_view
        self._max_view = max_view
        self._view_angle = view_angle
        self._grid = np.mgrid[0:self.shape[0]*self._scale:self._scale, 0:self.shape[1]*self._scale:self._scale]

        num_dilations = int(self._safety_buffer/self._scale)
        kernel = np.ones((3,3),np.uint8)
        map_dilated = cv2.dilate(self._img,kernel,iterations = num_dilations)
        self._safety_img = 0.25*self._img + 0.25*map_dilated
        self.all_waypoints = None
        self._frustum = got.defineFrustumPts(self._scale, self._min_view, self._max_view, self._view_angle)

        # gain on turning penalty for paths
        self._rho = rho
    #
    def generateNewMap(self, raw_file_name, output_file_name, bw_thresh, img_raw_scale, visualize = False):
        # TODO: Figure out the best way to do this this may need to be changed to get it to work with a new map
        map_raw = cv2.imread(raw_file_name,cv2.IMREAD_GRAYSCALE)
        if visualize:
            cv2.imshow('raw',map_raw)
        #
        map_bw = cv2.threshold(map_raw, bw_thresh, 255, cv2.THRESH_BINARY)[1]
        map_bw = cv2.bitwise_not(map_bw)
        if visualize:
            cv2.imshow('threshold_bw',map_bw)
        #
        # try to clean up noise in the map
        kernel = np.ones((5,5),np.uint8)
        map_bw = cv2.morphologyEx(map_bw, cv2.MORPH_CLOSE,kernel)
        map_bw = cv2.morphologyEx(map_bw, cv2.MORPH_CLOSE,kernel)
        if visualize:
            cv2.imshow('filtered', map_bw)
        #
        # shrink image to get desired scale
        scale_px2m = img_raw_scale #measured estimate for this case
        scale_des = self._scale
        height,width = map_bw.shape
        new_height = int(height*scale_px2m/scale_des)
        new_width = int(width*scale_px2m/scale_des)
        map_shrunk = cv2.resize(map_bw,(new_width,new_height))
        if visualize:
            cv2.imshow('resized',map_shrunk)
        #
        cv2.imwrite(output_file_name, map_shrunk)
        map_mat = np.array(map_shrunk)
        if visualize:
            # shut down when done
            k = cv2.waitKey(0)
            if k == 27:         # wait for ESC key to exit
                cv2.destroyAllWindows()
            #
        #
        self._img = map_mat
    #
    def getClosestObstacles(self, XY_scale):
        # now we need to move dots to be at least safety_buffer away from obstacles
        obstacles = (np.array(np.nonzero(self._img)) * self._scale)
        displacements = np.array([obstacles[None, 0] - XY_scale[:,0, None],
                                  obstacles[None, 1] - XY_scale[:,1, None]])
        #
        distances = np.linalg.norm(displacements, axis=0)
        angles = np.arctan2(displacements[1], displacements[0])

        closest = np.argmin(distances, axis=1)
        print(closest.shape)
        min_distances = distances[range(len(closest)),closest]
        min_angles = angles[range(len(closest)),closest]

        return min_distances, min_angles
    #
    def lineCollisionCheck(self, first, second, safety_buffer):
        # Uses Line Equation to check for collisions along new line made by connecting nodes
        x1 = first[0]
        y1 = first[1]
        x2 = second[0]
        y2 = second[1]

        a = y2 - y1
        b = x2 - x1
        c = x2*y1 - y2*x1
        if a == b and b == 0:
            return False
        #
        num_dilations = int(safety_buffer/self._scale)
        kernel = np.ones((3,3),np.uint8)
        map_dilated = cv2.dilate(self._img,kernel,iterations = num_dilations)
        obstacles = (np.array(np.nonzero(map_dilated))).T
        dist = abs(a*obstacles[:,0]-b*obstacles[:,1]+c)/np.sqrt(a*a+b*b)
        # filter to only look at obstacles within range of endpoints of lines
        prox = np.bitwise_not(np.bitwise_and(
                np.bitwise_or(
                    np.bitwise_and(obstacles[:,0]<=x2, obstacles[:,0]<=x1),
                    np.bitwise_and(obstacles[:,0]>=x2, obstacles[:,0]>=x1)),
                np.bitwise_or(
                    np.bitwise_and(obstacles[:,1]<=y2,obstacles[:,1]<=y1),
                    np.bitwise_and(obstacles[:,1]>=y2,obstacles[:,1]>=y1))))
        #
        if dist[prox].size > 0:
            if min(dist[prox])<=1:
                return False
            else:
                return True
            #
        else:
            return True
        #
    #
    def visualize(self):
        cv2.namedWindow('Map With Buffer')
        cv2.imshow('Map With Buffer',self._safety_img)
        cv2.waitKey()
        cv2.destroyWindow('Map With Buffer')
    #
    def visualizeWaypoints(self, waypoints, start_idx=None):
        pac_dots = np.zeros_like(self._img)
        pac_dots[waypoints[:,0], waypoints[:,1]] = 1

        img = pac_dots + self._safety_img
        img_color = img[...,None]*np.array([1, 1, 1])
        if start_idx is not None:
            cv2.circle(img_color, (waypoints[start_idx,1], waypoints[start_idx,0]), 5, (0,0,1))
        #
        cv2.namedWindow('Map With Waypoints')
        cv2.imshow('Map With Waypoints', img_color)
        cv2.waitKey()
        cv2.destroyWindow('Map With Waypoints')
    #
    def visualizePath(self, waypoints, path_idx, fig):
        # make this draw lines instead of points
        img = self._safety_img
        img_color = img[...,None]*np.array([1, 1, 1])
        path = waypoints[path_idx]
        path = np.fliplr(path)
        path = list(map(tuple,path))
        for ii in range(len(path)-1):
            cv2.line(img_color, path[ii],path[ii+1], (0,1,0),1)
        #
        # ax = fig.add_subplot(1,3,3)
        gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1])
        ax = plt.subplot(gs[1])
        ax.set_title('Selected Path')

        # Hide grid lines
        ax.grid(False)

        # Hide axes ticks
        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout()
        plt.imshow(img_color)
        plt.show()
    #
    def visualizePathWithCoverage(self, waypoints, path_idx, fig, coverage_map, loop_closures, combo_closures, coverage, travel_dist, nearest_point):
        # make this draw lines instead of points
        img = self._safety_img.copy()
        cov_img = coverage_map
        # img = coverage_map
        path_colors = [(1.0,0,0),(0,1.0,0),(0,0,1.0)]
        lc_colors = [(0.5,0.1,0),(0,0.5,0.1),(0.1,0,0.5)]

        img[img==0] += cov_img[img==0]
        img[img<0.3] += cov_img[img<0.3]
        img = np.clip(img,0,1.0)
        img_color = img[...,None]*np.array([1, 1, 1])
        for agent in range(len(path_idx)):
            path = waypoints[path_idx[agent][path_idx[agent]!=-1]]
            path = np.fliplr(path)
            path = list(map(tuple,path))
            for ii in range(len(path)-1):
                cv2.line(img_color, path[ii],path[ii+1], path_colors[agent%3],1)
        #
            for lc in loop_closures[agent]:
                cv2.line(img_color, tuple(np.flip(waypoints[lc[0]],0)), tuple(np.flip(waypoints[lc[1]],0)), lc_colors[agent%3], 1)
        #
        for lc in combo_closures:
            cv2.line(img_color, tuple(np.flip(waypoints[lc[0]],0)), tuple(np.flip(waypoints[lc[1]],0)), (1.0,1.0,0), 1)
        # ax = fig.add_subplot(1,3,3)
        gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1])
        ax = plt.subplot(gs[1])
        ax.set_title('Selected Path')

        # Hide grid lines
        ax.grid(False)

        # Hide axes ticks
        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout()
        plt.imshow(img_color)
        label = "Path #: " + str(nearest_point) + "\n Coverage: " + str(round(-coverage*100,2)) + "%\n Travel Distance: " + str(round(travel_dist*self._scale,2)) + "m"
        plt.xlabel(label)
        plt.show()
    #

    def computeFrustums(self, graph):
        frustum_graph = np.zeros((graph.shape[0],graph.shape[1],self._num_rays+2,2))
        traverse_angles = np.zeros_like(graph)
        traverse_dists = np.zeros_like(graph)
        obstacles = (np.array(np.nonzero(self._img)) * self._scale)
        alpha = self._view_angle/self._num_rays
        ray_angles = np.arange(self._num_rays)*alpha - self._view_angle/2.0
        for from_wpt,wp in tqdm(enumerate(graph), desc="Precomputing Frustums"):
        # for from_wpt,wp in enumerate(graph):
            for to_wpt,node in enumerate(wp):
                if node == 1:
                    wpt_loc = self.all_waypoints[from_wpt]
                    next_wpt = self.all_waypoints[to_wpt]
                    wpt_theta = np.arctan2(next_wpt[1]-wpt_loc[1], next_wpt[0]-wpt_loc[0])
                    travel_cost = np.linalg.norm([next_wpt[1]-wpt_loc[1], next_wpt[0]-wpt_loc[0]])

                    # find relative position of all obstacles
                    displacements = np.array([obstacles[0] - wpt_loc[0]*self._scale,
                                              obstacles[1] - wpt_loc[1]*self._scale])

                    distances = np.linalg.norm(displacements, axis=0)
                    dist_thresh = distances<self._max_view
                    distances = distances[dist_thresh]
                    displacements = displacements[:,dist_thresh]
                    #
                    angles = np.arctan2(displacements[1], displacements[0]) - wpt_theta
                    # #wrap
                    angles[angles < -np.pi] += 2*np.pi
                    angles[angles >  np.pi] -= 2*np.pi
                    #
                    in_bounds = np.array(np.where(np.logical_and(angles<(self._view_angle/2),angles>-(self._view_angle/2))))
                    angles = angles[in_bounds]
                    distances = distances[in_bounds]
                    dangles = np.digitize(angles,ray_angles)
                    z = np.vstack((self._max_view/self._scale*np.ones_like(ray_angles[None, :]), ray_angles[None, :]))
                    for ii in range(int(self._num_rays)):
                        try:
                            z[0,ii] = min(distances[dangles==ii+1])/self._scale
                        except:
                            pass

                    center = (int(wpt_loc[1]), int(wpt_loc[0]))
                    pts = np.array([np.sin(z[1])*z[0],np.cos(z[1])*z[0]])
                    pts = np.hstack((pts,np.array([[np.sin(z[1,-1])*self._min_view],
                                             [np.cos(z[1,-1])*self._min_view]]),
                                   np.array([[np.sin(z[1,0])*self._min_view],
                                             [np.cos(z[1,0])*self._min_view]])))
                    Rot = got.getRot2D(wpt_theta).T
                    frustum = np.around(Rot.dot(pts)).astype('int32').T

                    frustum_graph[from_wpt,to_wpt] = frustum
                    traverse_angles[from_wpt,to_wpt] = wpt_theta
                    traverse_dists[from_wpt,to_wpt] = travel_cost

        self._traverse_frustum = frustum_graph.astype(np.int32)
        self._traverse_angles = traverse_angles
        self._traverse_dists = traverse_dists

    def getCoverage(self, waypoints, return_map=False):

        if self.all_waypoints is None:
            raise ValueError('Map has no waypoints')

        cover_map = np.copy(self._img)
        draw_map = np.copy(cover_map)
        buffer_mask = np.logical_and(self._safety_img<0.3,self._safety_img>0)

        travel_cost = 0.0
        for agent in range(waypoints.shape[0]):
            prev_theta = 0
            for idx, wpt in enumerate(waypoints[agent]):
                if wpt == -1 or idx == len(waypoints[agent])-1:
                    break
                #
                wpt_loc = self.all_waypoints[wpt]
                wpt_theta = self._traverse_angles[wpt,waypoints[agent][idx+1]]
                delta_theta = got.rad_wrap_pi(wpt_theta - prev_theta)
                travel_cost += self._traverse_dists[wpt,waypoints[agent][idx+1]] + self._rho*abs(delta_theta)
                center = (int(wpt_loc[1]), int(wpt_loc[0]))
                frustum = self._traverse_frustum[wpt,waypoints[agent][idx+1]]
                cv2.fillPoly(draw_map, [frustum], 1, offset=center)

        #this line returns coverage of total pixel seen
        # coverage = (np.sum(draw_map) - self._num_occluded)/(draw_map.size - self._num_occluded)

        #this line returns coverage of buffer seen (more important for mapping)
        coverage = np.sum(draw_map[buffer_mask])/draw_map[buffer_mask].size

        # TODO: blend the 2 coverage types to favor buffer over total. but still incentivize total

        # minimize negative coverage and minimize travel distance
        if return_map:
            return -coverage, travel_cost, draw_map
        else:
            return -coverage, travel_cost

        #
    #

    def getSoloLoopClosures(self, waypoints, return_loop_close = False):
        sep_thresh = 40
        num_lcs = (np.zeros(waypoints.shape[0])).astype(int)
        solo_loop_closures = []
        for agent in range(waypoints.shape[0]):
            num_loop_close = 0
            wps = waypoints[agent][waypoints[agent]!=-1]
            wpt_sequence = np.array([wps,np.roll(wps,-1)]).T
            wpt_sequence = wpt_sequence[0:-1]

            #find where path passes through the same waypoints at different times
            unq, unq_idx, unq_cnt = np.unique(wpt_sequence, axis=0, return_inverse=True, return_counts=True)
            cnt_mask = unq_cnt > 1
            dup_ids = unq[cnt_mask]
            cnt_idx, = np.nonzero(cnt_mask)
            idx_mask = np.in1d(unq_idx, cnt_idx)
            idx_idx, = np.nonzero(idx_mask)
            srt_idx = np.argsort(unq_idx[idx_mask])
            dup_idx = np.split(idx_idx[srt_idx], np.cumsum(unq_cnt[cnt_mask])[:-1])

            #only count as loop closure if they are separated by at least sep_thresh
            if len(dup_ids > 0):
                for dup in dup_idx:
                    if abs(dup[0]-dup[1] > sep_thresh):
                        num_loop_close +=1
                    #
                    else:
                        pass
                    #
                #
            #
            num_lcs[agent] = num_loop_close
            if return_loop_close ==True:
                lc = []
                for dup in dup_idx:
                    lc.append(wpt_sequence[dup[0]])
                #
                solo_loop_closures.append(lc)


        if return_loop_close == False:
            return num_lcs
        #
        else:
            return num_lcs, solo_loop_closures
        #
    #
    def getCombLoopClosures(self, waypoints, return_loop_close = False):
        #TODO Combine combo and individual loop closure functions into a single function because it is mostly repetitious
        sep_thresh = 5
        num_lcs = (np.zeros(waypoints.shape[0])).astype(int)
        comb_loop_closures = []
        num_loop_close = 0
        path_splits = (np.zeros(waypoints.shape[0])).astype(int)
        # paths_sequence = []
        # for agent in range(waypoints.shape[0]):
        #     wps = waypoints[agent][waypoints[agent]!=-1]
        #     wpt_sequence = np.array([wps,np.roll(wps,-1)]).T
        #     wpt_sequence = wpt_sequence[0:-1]
        #     paths_sequence.append(wpt_sequence)
        # for agent in range(waypoints.shape[0]-1):
        #     for wp1,seq in enumerate(paths_sequence[agent]):
        #         are_equal = (paths_sequence[agent + 1] == seq)
        #         lc = np.logical_and(lc[:,0],lc[:,1])
        #         set_trace()
        wps = waypoints[waypoints!=-1]
        wpt_sequence = np.array([wps,np.roll(wps,-1)]).T
        # wpt_sequence = wpt_sequence[0:-1]
        path_split_idx = 0
        for agent in range(waypoints.shape[0]):
            path_split_idx += len(waypoints[agent][waypoints[agent]!=-1])
            path_splits[agent] = path_split_idx
        # set_trace()

        #find where path passes through the same waypoints at different times
        unq, unq_idx, unq_cnt = np.unique(wpt_sequence, axis=0, return_inverse=True, return_counts=True)
        cnt_mask = unq_cnt > 1
        dup_ids = unq[cnt_mask]
        cnt_idx, = np.nonzero(cnt_mask)
        idx_mask = np.in1d(unq_idx, cnt_idx)
        idx_idx, = np.nonzero(idx_mask)
        srt_idx = np.argsort(unq_idx[idx_mask])
        dup_idx = np.split(idx_idx[srt_idx], np.cumsum(unq_cnt[cnt_mask])[:-1])
        if len(dup_ids > 0):
            for dup in dup_idx:
                if not np.isin(path_splits,dup).any():
                    agent_lc = np.digitize(dup,path_splits)
                    if len(np.unique(agent_lc))>1:
                        num_loop_close += 1
                        comb_loop_closures.append(wpt_sequence[dup][0])

        return num_loop_close, comb_loop_closures

        #
    #
#
