import numpy as np
import cv2

class Mappy(object):
    def __init__(self, img, scale, hall_width, safety_buffer):
        self._img = img
        self._scale = scale
        self._hall_width = hall_width
        self._safety_buffer = hall_width/2#safety_buffer
        self.shape = self._img.shape

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

        try:
            a = y2 - y1
            b = x2 - x1
            c = x2*y1 - y2*x1
        except ZeroDivisionError:
            return False
        if a == b and b == c and c == 0:
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

    def visualize_waypoints(self, waypoints):
        pac_dots = np.zeros_like(self._img)
        pac_dots[waypoints[:,0], waypoints[:,1]] = 1
        # self.pac_dots = self.pac_dots*(1-self._mappy)
        img = pac_dots + self._safety_img
        cv2.cv2.imshow('map with waypoints', img)
        cv2.waitKey()

    def visualize_path(self, path):
        pac_dots = np.zeros_like(self._img)
        # make this draw lines instead of points
        pac_dots[waypoints[:,0], waypoints[:,1]] = 1
        # self.pac_dots = self.pac_dots*(1-self._mappy)
        img = pac_dots + 0.25*self._mappy + 0.25*map_dilated
        cv2.cv2.imshow('map with waypoints', img)
        cv2.waitKey()
