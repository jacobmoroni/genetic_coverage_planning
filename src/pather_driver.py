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
import os

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

cwd = os.getcwd() #current working directory

###############################################################################
# parameters
###############################################################################
# setting this to true will use waypoints and traversability that have already been genrated
# set to false to re-generate waypoints and traversability graph for new map or altered parameters
use_old_graph = True
old_graph_fname = cwd + '/data/wilk_3_graph.npy'
old_wpts_fname = cwd + '/data/wilk_3_wpts.npy'

# file where the pre-scaled map is.
scaled_map_file = cwd + "/data/map_scaled.png"
raw_map_file = cwd + "/data/map.png"

# Map parameters
scale = 0.15 #scale from pixels to meters of scaled map
narrowest_hall = 1.75 #width in meters of narrowest_hall
num_rays = 15 #number of rays used to compute coverage with obstacles
min_view = 0.5 #minimum distance the camera counts as viewed
max_view = 7 #maximum distance the camera counts as viewed
view_angle = 69.4*np.pi/180 #horizontal field of view of camera (in radians)
rho = 0. #turning penalty gain. currently disabled
solo_sep_thresh = 10 #threshold for separation between loop closures
bw_thresh = 90 # value used to threshold a new map when converting to black and white
scale_px2m = 1/0.44*0.0254 #measured estimate of original image pixels to meters scale
coverage_blend = 1.0 #percentage of coverage score based on viewing walls vs all area

#Path parameters
path_memory = 10 #Memory of path where path generator will not return to unless no other option exists
max_traverse_dist = 3.5 #max distance traversable with one step

#Genetic Algorithm Parameters
gen_size = 100 #number of organisms per generation (must be even)
starting_path_len = 75 #length of initial path
num_agents = 5 #number of agents
gamma = 0.5 #roulette exponent >=0. 0 means no fitness pressure
coverage_constr_0 = 0.3 #starting coverage constraint
coverage_constr_f = 0.8 #final coverage constraint
coverage_aging = 60 #number of generations to age coverage constraint

#Organism Parameters
start_idx = [207,207,1,10,305,207] #waypoint index where all paths will begin
max_dna_len = 100 #maximum number of waypoints in a path
min_dna_len = 30 #minimimum number of waypoints in a path
crossover_prob = 0.7 #probability of performing crossover when generating new organisms
crossover_time_thresh = 70 #how close crossover points need to be to eachother to be considered
mutation_prob = 0.3 #probability of performing mutation on new organisms
muterpolate_prob = 0.3 #probability of performing muterpolation on new organisms
num_muterpolations = 20 #number of possible points to perform muterpolation
muterpolation_srch_dist = 5 #how far ahead to look from each point when performing muterpolation
muterpolation_sub_prob = 0.8 #probability of accepting muterpolation point
min_solo_lcs = 2 #minimum number of loop closures each agent must have with their own path
min_comb_lcs = 4 #minimum number of loop closures agents must have with other agents
flight_time_scale = 0.0001 #scaling factor for flight time used in maximin fitnesses

###############################################################################
map_params = {'scale':scale,
              'narrowest_hall':narrowest_hall,
              'num_rays':num_rays,
              'min_view':min_view,
              'max_view':max_view,
              'view_angle':view_angle,
              'rho':rho,
              'solo_sep_thresh':solo_sep_thresh,
              'bw_thresh':bw_thresh,
              'scale_px2m':scale_px2m,
              'coverage_blend':coverage_blend}

path_params = {'path_memory':path_memory,
               'max_traverse_dist':max_traverse_dist}

org_params = {'start_idx':start_idx,
              'max_dna_len':max_dna_len,
              'min_dna_len':min_dna_len,
              'crossover_prob':crossover_prob,
              'crossover_time_thresh':crossover_time_thresh,
              'mutation_prob':mutation_prob,
              'muterpolate_prob':muterpolate_prob,
              'num_muterpolations':num_muterpolations,
              'muterpolation_srch_dist':muterpolation_srch_dist,
              'muterpolation_sub_prob':muterpolation_sub_prob,
              'min_solo_lcs':min_solo_lcs,
              'min_comb_lcs':min_comb_lcs,
              'flight_time_scale':flight_time_scale}

gen_params = {'gen_size':gen_size,
              'starting_path_len':starting_path_len,
              'num_agents':num_agents,
              'gamma':gamma,
              'coverage_constr_0':coverage_constr_0,
              'coverage_constr_f':coverage_constr_f,
              'coverage_aging':coverage_aging,
              'org_params':org_params}

map_scaled = cv2.imread(scaled_map_file,cv2.IMREAD_GRAYSCALE)/255
map_raw = cv2.imread(raw_map_file,cv2.IMREAD_GRAYSCALE)

print("Generating map from image file.")
if use_old_graph:
    mappy = Mappy(map_scaled, map_params)
    pather = PathMaker(mappy, path_params)

    print("Loading waypoints and traversible graph from file.")
    pather.loadTraversableGraph(old_graph_fname)
    pather.loadWptsXY(old_wpts_fname)
    mappy.saveMap(cwd + 'data/map_scaled_new.png')

else:
    mappy = Mappy(map_raw, map_params, new_map=True)
    pather = PathMaker(mappy, path_params)

    print("Generating possible waypoints.")
    pather.smartlyPlaceDots()
    print("Check waypoints placed and starting location(s). Press ESC to continue")
    mappy.visualizeWaypoints(pather.waypoint_locs, start_idx)
    print("Generating traversible graph.")
    pather.computeTraversableGraph()
    print("new files have been saved. overwrite old files with new to use in future")
    pather.saveTraversableGraph(cwd + '/data/wilk_3_graph_new.npy')
    pather.saveWptsXY(cwd + '/data/wilk_3_wptsXY_new.npy')
#
mappy.all_waypoints = pather.waypoint_locs
print("Precomputing coverage map.")
mappy.computeFrustums(pather._graph)

print("Spawning the contestants.")
population = GeneticalGorithm(mappy, pather, gen_params)

pareto_hist = []
pareto_hist += [got.getObjValsList(population)]

got.plotty(population, pather, mappy)
pareto_hist += population.runEvolution(100)
