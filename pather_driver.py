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
from geneticalgorithm import GeneticalGorithm
from matplotlib import pyplot as plt
plt.ion()

def plotty(population):
    objs = np.array([thing._obj_val for thing in population._gen_parent])

    plt.scatter(objs[:,0], objs[:,1])

    plt.show()
#

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
bw_thresh = 90
scale_px2m = 1/0.44*0.0254 #measured estimate for this case

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

mappy.all_waypoints = pather.waypoint_locs
#
# for i in range(50):
# path_idx = pather.makeMeAPath(200,start_idx)
#     # mappy.visualizePath(pather._XY, path_idx)
#
# path_idx2 = pather.makeMeAPath(200,start_idx)
# # mappy.visualizeWaypoints(pather._XY, start_idx)
# mappy.getCoverage(path_idx)
# # set_trace()
#
# poppy = Organism(path_idx, mappy, scale, narrowest_hall, max_dna_len, pather)
# mommy = Organism(path_idx2, mappy, scale, narrowest_hall, max_dna_len, pather)
#
# poppy.crossover(mommy)
# poppy.mutation()

population = GeneticalGorithm( mappy, scale, narrowest_hall, max_dna_len, pather )

plotty(population)
population.runEvolution(5)


#
