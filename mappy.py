from IPython.core.debugger import set_trace
from importlib import reload

import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import gridspec

def getRot2D(theta):
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta),  np.cos(theta)]])
#
def defineFrustumPts(scale, min_view, max_view, view_angle):
    num_rays = 10.
    # num_rays = 1.
    # alpha: width of the ray
    alpha = view_angle/num_rays
    ray_angles = np.arange(num_rays)*alpha - view_angle/2.
    pts = []
    # define points on the inside of the view
    for angle in ray_angles:
        rot = getRot2D(-angle)
        pts.append(rot.dot([0, min_view/scale]))
    #
    # define points on the outside of the view
    for angle in ray_angles:
        rot = getRot2D(angle)
        pts.append(rot.dot([0, max_view/scale]))
    #
    return np.array(pts)
#

def deg_wrap_180( angle ):
    angle -= 360.0 * np.floor((angle + 180.) * inv_360)
    return angle
#
def deg_wrap_360( angle ):
    angle -= 360.0 * np.floor(angle * inv_360)
    return angle
#
def rad_wrap_pi( angle ):
    angle -= 2*np.pi * np.floor((angle + np.pi) * inv_2pi)
    return angle
#
def rad_wrap_2pi( angle ):
    angle -= 2*np.pi * np.floor(angle * inv_2pi)
    return angle
#

class Mappy(object):
    def __init__(self, img, scale, hall_width, min_view, max_view, view_angle):
        self._img = img
        self._num_occluded = np.sum(self._img)
        self._scale = scale
        self._hall_width = hall_width
        self._safety_buffer = hall_width/2
        self.shape = self._img.shape
        # view area stuff for coverage calcualtion
        self._min_view = min_view
        self._max_view = max_view
        self._view_angle = view_angle/2
        self._grid = np.mgrid[0:self.shape[0]*self._scale:self._scale, 0:self.shape[1]*self._scale:self._scale]

        num_dilations = int(self._safety_buffer/self._scale)
        kernel = np.ones((3,3),np.uint8)
        map_dilated = cv2.dilate(self._img,kernel,iterations = num_dilations)
        self._safety_img = 0.25*self._img + 0.25*map_dilated
        self.all_waypoints = None
        self._frustum = defineFrustumPts(self._scale, self._min_view, self._max_view, self._view_angle)
    #
    def generateNewMap(self, raw_file_name, output_file_name, bw_thresh, img_raw_scale, visualize = False):
        # TODO: Figure out the best way to do this
        map_raw = cv2.imread(raw_file_name,cv2.IMREAD_GRAYSCALE)
        if visualize:
            cv2.imshow('raw',map_raw)
        #
        map_bw = cv2.threshold(map_raw, bw_thresh, 255, cv2.THRESH_BINARY)[1]
        map_bw = cv2.bitwise_not(map_bw)
        if visualize:
            cv2.imshow ('threshold_bw',map_bw)
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
        #
    #
    def visualize(self):
        cv2.namedWindow('Map With Buffer')
        cv2.imshow('Map With Buffer',self._safety_img)
        cv2.waitKey()
        cv2.destroyWindow('Map With Buffer')

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
        # cv2.namedWindow('Map With Path')
        # cv2.imshow('Map With Path', img_color)
        # cv2.waitKey()
        # cv2.destroyWindow('Map With Path')
    #
    def getCoverage(self, waypoints):
        if self.all_waypoints is None:
            raise ValueError('Map has no waypoints')
        # coverage = np.copy(self._img)
        # num_rays = int(self._max_view*self._view_angle/self._scale + 1)/2.
        # print(num_rays)
        num_rays = 10.
        # num_rays = 1.
        # alpha: width of the ray
        alpha = self._view_angle/num_rays
        ray_angles = np.arange(num_rays)*alpha - self._view_angle/2.
        # print(ray_angles)
        cover_map = np.copy(self._img)
        draw_map = np.copy(cover_map)

        travel_dist = 0.0
        for ii, wpt in enumerate(waypoints):
            if wpt == -1 or ii == len(waypoints)-1:
                break
            #
            wpt_loc = self.all_waypoints[wpt]
            next_wpt = self.all_waypoints[waypoints[ii+1]]
            wpt_theta = np.arctan2(next_wpt[1]-wpt_loc[1], next_wpt[0]-wpt_loc[0])
            travel_dist += np.linalg.norm([next_wpt[1]-wpt_loc[1], next_wpt[0]-wpt_loc[0]])

            # find relative position of all obstacles
            # obstacles = (np.array(np.nonzero(self._img)) * self._scale)
            # displacements = np.array([obstacles[None, 0] - wpt[0, None],
            #                           obstacles[None, 1] - wpt[1, None]])
            # distances = np.linalg.norm(displacements, axis=0)
            # angles = np.arctan2(displacements[1], displacements[0]) - wpt[2]
            # #wrap
            # angles[angles < -np.pi] += 2*np.pi
            # angles[angles >  np.pi] -= 2*np.pi

            # TODO make this actual measurements to obstacles
            z = np.vstack((self._max_view*np.ones_like(ray_angles[None, :]), ray_angles[None, :]))
            center = (int(wpt_loc[1]), int(wpt_loc[0]))
            pts = np.array([[-1, 1], [1, 1], [1, -1], [-1, -1]], dtype='int32')
            # rotate the frustum to the current heading
            Rot = getRot2D(wpt_theta).T
            view_mask = np.zeros_like(draw_map)

            # the collision checking starts getting really slow
            # cv2.fillPoly(view_mask, [np.around(Rot.dot(self._frustum.T)).astype('int32').T], 1, offset=center)
            # collisions = np.where(draw_map*view_mask)
            # draw_map = draw_map + view_mask - draw_map*view_mask

            cv2.fillPoly(draw_map, [np.around(Rot.dot(self._frustum.T)).astype('int32').T], 1, offset=center)

            #
        #
        coverage = (np.sum(draw_map) - self._num_occluded)/(draw_map.size - self._num_occluded)
        # print(coverage)
        # cv2.imshow("Drawn Coverage", draw_map)
        # cv2.waitKey()

        # minimize negative coverage and minimize travel distance
        return -coverage, travel_dist
    #
#
