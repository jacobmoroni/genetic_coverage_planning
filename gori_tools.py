from IPython.core.debugger import set_trace
from importlib import reload

import numpy as np
import cv2
import copy
import matplotlib.pyplot as plt
from matplotlib import gridspec
from tqdm import tqdm
import pointselector
reload(pointselector)
from pointselector import PointSelector



# ======================================
# ======================================

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

# ======================================
# ======================================

inv_360 = 1 / 360
inv_2pi = 0.5 / np.pi

def deg_wrap_180( angle ):
    angle -= 360.0 * np.floor((angle + 180.) * inv_360)
    return angle
#
def deg_wrap_360( angle ):
    angle -= 360.0 * np.floor(angle * inv_360)
    return angle
#
def rad_wrap_pi( angle ):
    angle -= 2*np.pi * np.floor((angle + np.pi) * inv_2pi)
    return angle
#
def rad_wrap_2pi( angle ):
    angle -= 2*np.pi * np.floor(angle * inv_2pi)
    return angle
#

# ======================================
# ======================================

def getRot2D(theta):
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta),  np.cos(theta)]])
#
def defineFrustumPts(scale, min_view, max_view, view_angle):
    num_rays = 10.
    # num_rays = 1.
    # alpha: width of the ray
    alpha = view_angle/num_rays
    ray_angles = np.arange(num_rays)*alpha - view_angle/2.
    pts = []
    # define points on the inside of the view
    for angle in ray_angles:
        rot = getRot2D(-angle)
        pts.append(rot.dot([0, min_view/scale]))
    #
    # define points on the outside of the view
    for angle in ray_angles:
        rot = getRot2D(angle)
        pts.append(rot.dot([0, max_view/scale]))
    #
    return np.array(pts)
#

# ======================================
# ======================================

def lowVarSample(X, fitnesses, pressure):
    # pressure should be between 0 and 1
    log_w = -np.array(fitnesses)
    log_w = log_w - np.max(log_w)
    base = pressure + 1
    w = base**log_w
    w = w/np.sum(w)
    Xbar = []
    M = len(X)
    r = np.random.uniform(0, 1/M)
    c = w[0]
    i = 0
    last_i = i
    for m in range(M):
        u = r + m/M
        while u > c:
            i += 1
            c = c + w[i]
        #
        new_x = copy.deepcopy(X[i])
        Xbar.append(new_x)
        last_i = i
    #
    return Xbar
#

def sortBy(stuff, stuff_values):
    order = np.argsort(stuff_values)
    stuff = list(np.array(stuff)[order])
    stuff_values = list(np.array(stuff_values)[order])
    return stuff, stuff_values
#

# ======================================
# ======================================

def giveMeOptions(graph, pt1, pt2):
    options = np.where(graph[pt1]*graph[pt2])
    return options
#

#
