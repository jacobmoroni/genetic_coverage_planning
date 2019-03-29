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
import pointselector
reload(pathmaker)
reload(mappy)
reload(geneticalgorithm)
reload(pointselector)
from pathmaker import PathMaker
from mappy import Mappy
from geneticalgorithm import GeneticalGorithm
from pointselector import PointSelector
from matplotlib import pyplot as plt
from matplotlib import gridspec

# plt.ion()

def plotty(population,pather,mappy):
    objs = np.array([thing._obj_val for thing in population._gen_parent])
    fig = plt.figure()
    gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1])
    ax = plt.subplot(gs[0])
    ax.set_title('Pareto Front')
    plt.ylabel('Flight Time (Cheetos)')
    plt.xlabel('-Coverage')
    plt.tight_layout()
    point, = ax.plot(objs[0,0],objs[0,1],'xr')
    pointy = PointSelector(point,objs,mappy,pather,population,fig)
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
print("Generating map from image file.")
mappy = Mappy(map_scaled, scale, narrowest_hall, min_view, max_view, view_angle)

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

mappy.all_waypoints = pather.waypoint_locs

print("Spawning the contestants.")
population = GeneticalGorithm( mappy, scale, narrowest_hall, max_dna_len, pather )

plotty(population, pather, mappy)
population.runEvolution(5)


#
