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
from geneticalgorithm import Organism

use_old_graph = True
old_graph_fname = 'wilk_3_graph.npy'
old_wpts_fname = 'wilk_3_wpts.npy'

file_name = "map_scaled.png"
scale = 0.15
narrowest_hall = 1.75
safety_buffer = 0.7
min_view = 0.5
max_view = 7
view_angle = 69.4*np.pi/180
max_dna_len = 250
start_idx = 207

map_scaled = cv2.imread(file_name,cv2.IMREAD_GRAYSCALE)/255

mappy = Mappy(map_scaled, scale, narrowest_hall, min_view, max_view, view_angle)

pather = PathMaker(mappy, scale, narrowest_hall, safety_buffer)

if use_old_graph:
    pather.loadTraversableGraph(old_graph_fname)
    pather.loadWptsXY(old_wpts_fname)
else:
    pather.smartlyPlaceDots()
    pather.computeTraversableGraph(3.5)
    pather.saveTraversableGraph('wilk_3_graph_new.npy')
    pather.saveWptsXY('wilk_3_wptsXY_new.npy')
#
# for i in range(2):
path_idx = pather.makeMeAPath(200,start_idx)
path_idx2 = pather.makeMeAPath(200,start_idx)
    # mappy.visualize_path(pather._XY, path_idx)
# mappy.visualize_waypoints(pather._XY, start_idx)
mappy.getCoverage(pather._XY, path_idx)
# set_trace()

poppy = Organism(path_idx, mappy, scale, narrowest_hall, max_dna_len, pather)
mommy = Organism(path_idx2, mappy, scale, narrowest_hall, max_dna_len, pather)

poppy.crossover(mommy)
poppy.mutation()

#
