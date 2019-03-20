#this removes python2.7 paths so it wont screw everything up
import sys
dir_remove = []
for p in sys.path:
    if p.find('python2') !=-1:
        dir_remove.append(p)
for p in dir_remove:
    sys.path.remove(p)

import numpy as np
import mapmaker
import fov_frustrum

file_name = '/home/jacob/byu_classes/optimization/genetic_coverage_planning/map.png'
thresh = 90
scale_px2m = 1/0.44*0.0254 #measured estimate for this case 
scale_des = 0.15 #roughly 6inches per pixel

bot_location = np.array([0,0])
bot_angle = np.deg2rad(0)
min_d = 0.5
max_d = 6.0

[obstacles, free_space] = generateMap(file_name,thresh,scale_px2m, scale_des, False)
frustrum = getFrustrumPoints(bot_location,bot_angle,min_d,max_d)
print("this")
