#! /usr/bin/env python3
import sys
dir_remove = []
for p in sys.path:
    if p.find('python2') !=-1:
        dir_remove.append(p)
    #
#
for p in dir_remove:
    sys.path.remove(p)
#
from IPython.core.debugger import set_trace
from importlib import reload

import numpy as np
import cv2

from matplotlib import pyplot as plt
from matplotlib import gridspec
# plt.ion()

import gori_tools as got
import pathmaker
import mappy
import geneticalgorithm
import pointselector
reload(got)
reload(pathmaker)
reload(mappy)
reload(geneticalgorithm)
reload(pointselector)
from pathmaker import PathMaker
from mappy import Mappy
from geneticalgorithm import GeneticalGorithm
from pointselector import PointSelector


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
rho = 0.


map_scaled = cv2.imread(file_name,cv2.IMREAD_GRAYSCALE)/255
print("Generating map from image file.")
mappy = Mappy(map_scaled, scale, narrowest_hall, min_view, max_view, view_angle, rho)

pather = PathMaker(mappy, scale, narrowest_hall, safety_buffer)

if use_old_graph:
    print("Loading waypoints and traversible graph from file.")
    pather.loadTraversableGraph(old_graph_fname)
    pather.loadWptsXY(old_wpts_fname)
else:
    print("Generating possible waypoints.")
    pather.smartlyPlaceDots()
    print("Generating traversible graph.")
    pather.computeTraversableGraph(3.5)
    pather.saveTraversableGraph('wilk_3_graph_new.npy')
    pather.saveWptsXY('wilk_3_wptsXY_new.npy')
#
mappy.all_waypoints = pather.waypoint_locs

print("Spawning the contestants.")
population = GeneticalGorithm( mappy, scale, narrowest_hall, max_dna_len, pather )

pareto_hist = []
# pareto_hist.extend( got.getObjValsList(population) )
pareto_hist += [got.getObjValsList(population)]

got.plotty(population, pather, mappy)
# for ii in range(2):
pareto_hist += population.runEvolution(100)
#

#
