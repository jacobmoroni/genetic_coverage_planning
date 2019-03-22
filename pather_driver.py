#! /usr/bin/env python3
import sys
dir_remove = []
for p in sys.path:
    if p.find('python2') !=-1:
        dir_remove.append(p)
for p in dir_remove:
    sys.path.remove(p)

from IPython.core.debugger import set_trace
from importlib import reload

import numpy as np
import cv2

import pathmaker
import mappy
import geneticalgorithm
reload(pathmaker)
reload(mappy)
reload(geneticalgorithm)
from pathmaker import PathMaker
from mappy import Mappy
from geneticalgorithm import Chromosome

file_name = "map_scaled.png"
scale = 0.15
narrowest_hall = 1.75
safety_buffer = 0.7
min_view = 0.5
max_view = 7
view_angle = 69.4*np.pi/180
max_chromo_len = 250
start_idx = 207

map_scaled = cv2.imread(file_name,cv2.IMREAD_GRAYSCALE)/255

mappy = Mappy(map_scaled, scale, narrowest_hall, min_view, max_view, view_angle)

pather = PathMaker(mappy, scale, narrowest_hall, safety_buffer)

pather.smartly_place_dots()
pather.compute_traversable_graph(3.5)
# for i in range(2):
path_idx = pather.makeMeAPath(200,start_idx,5)
path_idx2 = pather.makeMeAPath(200,start_idx,5)
    # mappy.visualize_path(pather._XY, path_idx)
# mappy.visualize_waypoints(pather._XY, start_idx)

poppy = Chromosome(path_idx, mappy, scale, narrowest_hall, max_chromo_len)
mommy = Chromosome(path_idx2, mappy, scale, narrowest_hall, max_chromo_len)

# set_trace()
poppy.crossover(mommy)

#
