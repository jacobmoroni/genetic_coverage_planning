#this removes python2.7 paths so it wont screw everything up

import sys
dir_remove = []
for p in sys.path:
    if p.find('python2') !=-1:
        dir_remove.append(p)
for p in dir_remove:
    sys.path.remove(p)
#

from IPython.core.debugger import set_trace
from importlib import reload

import numpy as np
import cv2
import matplotlib.pyplot as plt

file_name = '/home/jacob/byu_classes/optimization/genetic_coverage_planning/map.png'
# file_name = '/home/jacob/Documents/byu_classes/optimization/genetic_coverage_planning/map.png'
bw_thresh = 90
scale_px2m = 1/0.44*0.0254 #measured estimate for this case
scale_des = 0.15 #roughly 6inches per pixel

def generateMap(file_name, bw_thresh, img_scale, target_scale, visualize):
    # set_trace()
    map_raw = cv2.imread(file_name,cv2.IMREAD_GRAYSCALE)
    if visualize:
        cv2.imshow('raw',map_raw)

    map_bw = cv2.threshold(map_raw, bw_thresh, 255, cv2.THRESH_BINARY)[1]
    map_bw = cv2.bitwise_not(map_bw)
    if visualize:
        cv2.imshow ('threshold_bw',map_bw)

    #try to clean up noise in the map
    kernel = np.ones((5,5),np.uint8)
    map_bw = cv2.morphologyEx(map_bw, cv2.MORPH_CLOSE,kernel)
    map_bw = cv2.morphologyEx(map_bw, cv2.MORPH_CLOSE,kernel)
    if visualize:
        cv2.imshow('filtered', map_bw)

    #shrink image to get desired scale
    scale_px2m = img_scale #measured estimate for this case
    scale_des = target_scale #roughly 6inches per pixel
    height,width = map_bw.shape
    new_height = int(height*scale_px2m/scale_des)
    new_width = int(width*scale_px2m/scale_des)
    map_shrunk = cv2.resize(map_bw,(new_width,new_height))
    if visualize:
        cv2.imshow('resized',map_shrunk)

        cv2.imwrite('map_scaled.png',map_shrunk)
    #convert to array to begin generating obstacles
    map_mat = np.array(map_shrunk)
    if visualize:
     #shut down when done
        k = cv2.waitKey(0)
        if k == 27:         #wait for ESC key to exit
            cv2.destroyAllWindows()
        elif k == ord('s'):  #wait for 's' key to save and exit

            cv2.destroyAllWindows()

    return map_mat

#these functions are used to generate paths through map. with no predetermined waypoints.

def rot2d(angle):
    return np.array([[np.cos(angle), -np.sin(angle)],[np.sin(angle),np.cos(angle)]])

def generateObsAndFree(map, scale_des):
    obs_px = np.array(np.nonzero(map))
    obs_m = obs_px*scale_des

    free_space_px = np.array(np.where(map == 0))
    free_space_m = free_space_px*(scale_des)

    return obs_m, free_space_m

def getDilatedMap(map, dilation, scale):
    num_dilations = int(dilation/scale)
    kernel = np.ones((3,3),np.uint8)
    map_dilated = cv2.dilate(map,kernel,iterations = num_dilations)
    cv2.imshow('dilated',map_dilated)
    k = cv2.waitKey(0)
    if k == 27:         #wait for ESC key to exit
        cv2.destroyAllWindows()
    elif k == ord('s'):  #wait for 's' key to save and exit
        # cv2.imwrite('messigray.png',img)
        cv2.destroyAllWindows()

    return map_dilated

def isFeasible(point, map, scale):
    # set_trace()
    if map[int(point[0]/scale),int(point[1]/scale)]== 0:
        return True
    else:
        return False

def generateFeasiblePath(num_steps, velocity, dt, map, start_point, scale):
    stuck_counter = 0
    step = 0
    theta = 0
    path = np.array([start_point])
    theta_path = np.array([theta])
    current_point = start_point
    while step < num_steps:
        theta_diff = np.random.randn(1)*np.pi/4
        theta_diff = theta_diff[0]
        next_point = current_point + np.array([velocity*dt,0])@rot2d(theta+theta_diff)
        if isFeasible(next_point,map,scale):
            path = np.append(path,np.array([next_point]),axis = 0)
            current_point = next_point
            theta = theta + theta_diff
            theta_path = np.append(theta_path, theta)
            step = step + 1
            stuck_counter = 0
            print (step)
        else:
            stuck_counter = stuck_counter + 1

        if stuck_counter > 50:
            print ("got stuck")
            return path, theta_path

    return path, theta_path


stuck_counter= 0

current_map = generateMap(file_name,bw_thresh,scale_px2m, scale_des,True)
dilated_map = getDilatedMap(current_map,0.7,scale_des)
obs, free = generateObsAndFree(dilated_map, scale_des)
start_point = np.array([23, 2])
path, theta_path = generateFeasiblePath(700, 1, 1,dilated_map,start_point, scale_des)
set_trace()

plt.plot(free[1],-free[0],'.')
plt.plot(start_point[1], -start_point[0],'rx')
plt.plot(path[:,1],-path[:,0],'r')
plt.show()
