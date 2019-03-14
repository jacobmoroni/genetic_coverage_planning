import numpy as np
import cv2
from pdb import set_trace
import matplotlib.pyplot as plt
# map_raw = cv2.imread('lab_map.png',1)
map_raw = cv2.imread('/home/jacob/Documents/byu_classes/optimization/genetic path planning/map.png',0)
cv2.imshow('raw',map_raw)

thresh = 90
map_bw = cv2.threshold(map_raw, thresh, 255, cv2.THRESH_BINARY)[1]
map_bw = cv2.bitwise_not(map_bw)
cv2.imshow ('threshold_bw',map_bw)

#try to clean up noise in the map
kernel = np.ones((5,5),np.uint8)
# map_bw = cv2.dilate(map_bw,kernel,iterations = 3)
# map_bw = cv2.erode(map_bw,kernel,iterations = 3)
map_bw = cv2.morphologyEx(map_bw, cv2.MORPH_CLOSE,kernel)
map_bw = cv2.morphologyEx(map_bw, cv2.MORPH_CLOSE,kernel)
# map_bw = cv2.morphologyEx(map_bw, cv2.MORPH_OPEN,kernel)
cv2.imshow('filtered', map_bw)

#shrink image to get desired scale
scale_px2m = 1/0.44*0.0254 #measured estimate for this case 
scale_des = 0.15 #roughly 6inches per pixel
height,width = map_bw.shape
new_height = int(height*scale_px2m/scale_des)
new_width = int(width*scale_px2m/scale_des)
map_shrunk = cv2.resize(map_bw,(new_width,new_height))
cv2.imshow('resized',map_shrunk)
cv2.waitKey()
#convert to array to begin generating obstacles
map_mat = np.array(map_shrunk)
obs = np.nonzero(map_mat)
obs = np.array(obs)

free_space_px = np.array(np.where(map_mat == 0))
free_space_m = free_space_px*(scale_des)
print(free_space_m.shape)
plt.plot(free_space_m[1],-free_space_m[0],'.')
plt.show()
print("done")
#prune obstacles
#TODO add something to pass in less obstacles 

#convert pixels to meters
# px_conv=0.03103
# ob_list = []
# prox_thresh = 0.1
# for i,x in enumerate(obs.T):
    # x = x*px_conv
    # if i == 0:
        # ob_list.append((x[0],x[1],0.11))
    # else:
        # min_dist = 1
        # for i in range(0,len(ob_list)):
            # min_dist =min(np.sqrt((x[0]-ob_list[i][0])**2+(x[1]-ob_list[i][1])**2),min_dist)
        # if min_dist > prox_thresh:
            # ob_list.append((x[0],x[1],0.11))
            # # print min_dist
            # # print len(ob_list)
        # else:
            # pass
            # # print "not added"
            # #TODO figure out how to do this loop to only add the obstacle if 
            # #no other obstacle already added is within the threshold


# print ob_list


# #shut down when done
k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()
elif k == ord('s'): # wait for 's' key to save and exit
    cv2.imwrite('messigray.png',img)
    cv2.destroyAllWindows()
