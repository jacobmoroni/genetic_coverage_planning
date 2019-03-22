from IPython.core.debugger import set_trace
from importlib import reload

import numpy as np
import cv2

class Mappy(object):
    def __init__(self, img, scale, hall_width, min_view, max_view, view_angle):
        self._img = img
        self._scale = scale
        self._hall_width = hall_width
        self._safety_buffer = hall_width/2#safety_buffer
        self.shape = self._img.shape
        # view area stuff for coverage calcualtion
        self._min_view = min_view
        self._max_view = max_view
        self._view_angle = view_angle
        self._grid = np.mgrid[0:self.shape[0] + self._scale:self._scale, 0:self.shape[0] + self._scale:self._scale]

        num_dilations = int(self._safety_buffer/self._scale)
        kernel = np.ones((3,3),np.uint8)
        map_dilated = cv2.dilate(self._img,kernel,iterations = num_dilations)
        self._safety_img = 0.25*self._img + 0.25*map_dilated

    def getClosestObstacles(self, XY_scale):
        # now we need to move dots to be at least safety_buffer away from obstacles
        obstacles = (np.array(np.nonzero(self._img)) * self._scale)
        displacements = np.array([obstacles[None, 0] - XY_scale[:,0, None],
                                  obstacles[None, 1] - XY_scale[:,1, None]])
        distances = np.linalg.norm(displacements, axis=0)
        angles = np.arctan2(displacements[1], displacements[0])

        closest = np.argmin(distances, axis=1)
        print(closest.shape)
        min_distances = distances[range(len(closest)),closest]
        min_angles = angles[range(len(closest)),closest]
        return min_distances, min_angles

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

        obstacles = (np.array(np.nonzero(self._img))).T
        dist = abs(a*obstacles[:,0]-b*obstacles[:,1]+c)/np.sqrt(a*a+b*b)-safety_buffer
        #filter to only look at obstacles within range of endpoints of lines
        prox = np.bitwise_not(np.bitwise_and(
                np.bitwise_or(
                    np.bitwise_and(obstacles[:,0]<=x2, obstacles[:,0]<=x1),
                    np.bitwise_and(obstacles[:,0]>=x2, obstacles[:,0]>=x1)),
                np.bitwise_or(
                    np.bitwise_and(obstacles[:,1]<=y2,obstacles[:,1]<=y1),
                    np.bitwise_and(obstacles[:,1]>=y2,obstacles[:,1]>=y1))))

        if dist[prox].size > 0:
            if min(dist[prox])<=0:
                return False
            else:
                return True



    def visualize(self):
        cv2.cv2.imshow('map with buffer',self._safety_img)
        cv2.waitKey()

    def visualize_waypoints(self, waypoints, start_idx=None):
        pac_dots = np.zeros_like(self._img)
        pac_dots[waypoints[:,0], waypoints[:,1]] = 1
        # self.pac_dots = self.pac_dots*(1-self._mappy)
        img = pac_dots + self._safety_img
        img_color = img[...,None]*np.array([1, 1, 1])
        if start_idx is not None:
            cv2.circle(img_color, (waypoints[start_idx,1], waypoints[start_idx,0]), 5, (0,0,1))
        cv2.cv2.imshow('map with waypoints', img_color)
        cv2.waitKey()

    def visualize_path(self, waypoints, path_idx):
        # make this draw lines instead of points
        img = self._safety_img
        img_color = img[...,None]*np.array([1, 1, 1])
        path = waypoints[path_idx]
        path = np.fliplr(path)
        path = list(map(tuple,path))
        for ii in range(len(path)-1):
            cv2.line(img_color, path[ii],path[ii+1], (0,1,0),1)
            # cv2.line(img, path[ii],path[ii+1], (1,1,1),1)

        # pac_dots[waypoints[:,0], waypoints[:,1]] = (0,1,0)
        # self.pac_dots = self.pac_dots*(1-self._mappy)
        cv2.imshow('map with path', img_color)
        # cv2.imshow('map with all path', img)
        cv2.waitKey()

    def get_coverage(self, waypoints):
        coverage = np.copy(self._img)
        num_rays = int(self._max_view*self._view_angle/self._scale + 1)
        ray_angles = np.arange(num_rays)*self._view_angle/num_rays - self._view_angle/(num_rays/2.)
        for wpt in waypoints:
            # find relative position of all obstacles
            # obstacles = (np.array(np.nonzero(self._img)) * self._scale)
            # displacements = np.array([obstacles[None, 0] - wpt[0, None],
            #                           obstacles[None, 1] - wpt[1, None]])
            # distances = np.linalg.norm(displacements, axis=0)
            # angles = np.arctan2(displacements[1], displacements[0]) - wpt[2]
            # #wrap
            # angles[angles < -np.pi] += 2*np.pi
            # angles[angles >  np.pi] -= 2*np.pi

            # trace several rays that simulate the FOV
            rel_grid = self._grid - x[:2, np.newaxis, np.newaxis]
    #         print(rel_grid.shape
            r_grid = np.linalg.norm(rel_grid, axis=0)
            theta_grid = np.arctan2(rel_grid[1, :, :], rel_grid[0, :, :]) - x[2]
            # wrap
            theta_grid[theta_grid < -np.pi] += 2*np.pi
            theta_grid[theta_grid >  np.pi] -= 2*np.pi



################################################################################
class Occ_Map(object):
    def __init__(self, width=100, height=100, resolution=1.0,
                 z_max = 150, alpha= 1., beta=2*np.pi/180,
                 p_free = 0.4, p_occ = 0.6):
        m_width = int(width/resolution)
        m_height = int(height/resolution)
        self.z_max = z_max
        self.alpha = alpha
        self.beta = beta
        self.l_free = np.log(p_free/(1 - p_free))
        self.l_occ = np.log(p_occ/(1 - p_occ))
#         self._m = np.zeros(m_width + 1, m_height + 1) + 0.5
        self._log_m = np.zeros((m_width + 1, m_height + 1))
        self._grid = np.mgrid[0:width + resolution:resolution, 0:height + resolution:resolution]

    def get_map(self):
        return 1.0 - 1.0/(1.0 + np.exp(self._log_m))

    def update(self, x, z, thk):
        rel_grid = self._grid - x[:2, np.newaxis, np.newaxis]
#         print(rel_grid.shape)

        r_grid = np.linalg.norm(rel_grid, axis=0)

        theta_grid = np.arctan2(rel_grid[1, :, :], rel_grid[0, :, :]) - x[2]
        # wrap
        theta_grid[theta_grid < -np.pi] += 2*np.pi
        theta_grid[theta_grid >  np.pi] -= 2*np.pi

        # generate an update mask
        meas_mask = r_grid[:, :, np.newaxis] < z[0, np.newaxis, np.newaxis, :] - self.alpha/2.

        max_mask = (r_grid < self.z_max)[:, :, np.newaxis]
        max_mask = np.tile(max_mask, (1, 1, z.shape[1]))
#         print("shape: {}".format(max_mask.shape))

        theta_mask = np.abs(theta_grid[:, :, np.newaxis] - thk[np.newaxis, :]) < self.beta/2.


        free_mask = np.logical_or.reduce(max_mask & theta_mask & meas_mask, axis=-1)
#         print("shape: {}".format(free_mask.shape))

        # mask out the measurement range +/- the thickness alpha
        meas_mask = r_grid[:, :, np.newaxis] < z[0, np.newaxis, np.newaxis, :] + self.alpha/2.
        meas_mask = meas_mask & (r_grid[:, :, np.newaxis] > z[0, np.newaxis, np.newaxis, :] - self.alpha/2.)

        occ_mask = np.logical_or.reduce(max_mask & theta_mask & meas_mask, axis=-1)
#         print("shape: {}".format(occ_mask.shape))
#         m_handle = plt.imshow(1- 1*np.flip(occ_mask[:, :].T, 0), cmap="gray")
#         plt.show()
#         print(rel_grid[:, :, 0])
#         print("shape: {}".format(theta_mask.shape))

        self._log_m += free_mask*self.l_free + occ_mask*self.l_occ
