import numpy as np
import cv2
from pdb import set_trace
import matplotlib.pyplot as plt

file_name = '/home/jacob/byu_classes/optimization/genetic_coverage_planning/map.png'
thresh = 90
scale_px2m = 1/0.44*0.0254 #measured estimate for this case 
scale_des = 0.15 #roughly 6inches per pixel

def generateMap(file_name,thresh, img_scale, target_scale, visualize):
    # set_trace()
    map_raw = cv2.imread(file_name,0)
    if visualize:
        cv2.imshow('raw',map_raw)

    map_bw = cv2.threshold(map_raw, thresh, 255, cv2.THRESH_BINARY)[1]
    # map_bw = cv2.bitwise_not(map_bw)
    if visualize:
        cv2.imshow ('threshold_bw',map_bw)

    #try to clean up noise in the map
    kernel = np.ones((5,5),np.uint8)
    # map_bw = cv2.dilate(map_bw,kernel,iterations = 3)
    # map_bw = cv2.erode(map_bw,kernel,iterations = 3)
    map_bw = cv2.morphologyEx(map_bw, cv2.MORPH_CLOSE,kernel)
    map_bw = cv2.morphologyEx(map_bw, cv2.MORPH_CLOSE,kernel)
    # map_bw = cv2.morphologyEx(map_bw, cv2.MORPH_OPEN,kernel)
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
        cv2.waitKey()
    #convert to array to begin generating obstacles
    map_mat = np.array(map_shrunk)
    if visualize:
     #shut down when done
        k = cv2.waitKey(0)
        if k == 27:         #wait for ESC key to exit
            cv2.destroyAllWindows()
        elif k == ord('s'):  #wait for 's' key to save and exit
            # cv2.imwrite('messigray.png',img)
            cv2.destroyAllWindows()

    return map_mat

def generateObsAndFree(map, scale_des):
    free_space_px = np.array(np.nonzero(map))
    free_space_m = free_space_px*(scale_des)

    obs_px = np.array(np.where(map == 0))
    obs_m = obs_px*scale_des

    plt.plot(free_space_m[1],-free_space_m[0],'.')
    plt.show()
    return obs_m, free_space_m

def getDilatedMap(map, dilation, scale):
    num_dilations = int(dilation/scale)
    kernel = np.ones((5,5),np.uint8)
    map_dilated = cv2.erode(map,kernel,iterations = num_dilations)
    cv2.imshow('dilated',map_dilated)
    cv2.waitKey()
    return map_dilated




current_map = generateMap(file_name,thresh,scale_px2m, scale_des, False)
dilated_map = getDilatedMap(current_map,1.0,scale_des)

