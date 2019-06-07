from IPython.core.debugger import set_trace
from importlib import reload

import numpy as np
import cv2.cv2 as cv2
import copy
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.animation as animation
import pickle
import math
from functools import reduce

from tqdm import tqdm
import pointselector
reload(pointselector)
from pointselector import PointSelector

# ==============================================================================
# ==============================================================================

def plotty(population,pather,mappy):
    objs = np.array([thing._obj_val for thing in population._gen_parent])
    fig = plt.figure()
    gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1])
    ax = plt.subplot(gs[0])
    ax.set_title('Pareto Front')
    plt.ylabel('Flight Time')
    plt.xlabel('-Coverage')
    plt.tight_layout()
    point, = ax.plot(objs[0,0],objs[0,1],'xr')
    PointSelector(point,objs,mappy,pather,population,fig)
    plt.plot(objs[:,0], objs[:,1],'.b')
    plt.show()

def animateFlight(current_organism, waypoints, cov_img, img):
    img[img==0] += cov_img[img==0]
    img[img<0.3] += cov_img[img<0.3]
    img = np.clip(img,0,1.0)
    img_color = img[...,None]*np.array([1, 1, 1])

    paths = current_organism._dna

    path_colors = [(1.0,0,0),
                     (0,1.0,0),
                     (0,0,1.0),
                     (0.8,0.8,0.0),
                     (0.0,1.0,1.0),
                     (1.0,0.0,1.0)]
    num_colors = 6
    max_path_len = current_organism._max_dna_len
    num_agents = int(paths.shape[0])
    last_point = np.zeros(num_agents).astype(int)
    combo_path = np.zeros((num_agents,max_path_len,2))
    end_of_path = np.zeros(num_agents)
    for agent in range(num_agents):
        path = waypoints[paths[agent]]
        path = np.fliplr(path)
        combo_path[agent] = path
    combo_path = combo_path.astype(int)
    for wp in range(max_path_len-1):
        plt.cla()
        plt.xticks([])
        plt.yticks([])
        for agent in range(num_agents):
            if paths[agent,wp+1] != -1:
                cv2.line(img_color,
                         tuple(combo_path[agent,wp]),
                         tuple(combo_path[agent,wp+1]),
                         path_colors[agent % num_colors],
                         1)
                last_point[agent] = wp
            else:
                cv2.line(img_color,
                         tuple(combo_path[agent,last_point[agent]]),
                         tuple(combo_path[agent,last_point[agent]+1]),
                         path_colors[agent % num_colors],
                         1)
                end_of_path[agent] = True
        if end_of_path.all():
            break

        plt.imshow(img_color)
        for agent in range(num_agents):
            if paths[agent,wp+1] != -1:
                plt.plot(combo_path[agent, wp+1, 0],
                         combo_path[agent, wp+1, 1],
                         'ok')
            else:
                plt.plot(combo_path[agent, last_point[agent]+1, 0],
                         combo_path[agent, last_point[agent]+1, 1],
                         'ok')

        plt.pause(0.01)

def update_pareto(gen_num, data, population):
    population.set_data(data[..., gen_num])
    plt.xlim(-1,max(data[0,:,gen_num])+0.05)
    plt.ylim(0,max(data[1,:,gen_num])+0.1)
    # text = "generation number: "+str(gen_num)
    # plt.text(-1,0.38,text, fontsize=14)
    return population,

def plotParetoHist(pareto_hist):
    pareto_hist = np.array(pareto_hist)
    fig1= plt.figure()
    num_gen = pareto_hist.shape[0]
    data = np.transpose(pareto_hist, (2,1,0))
    l, = plt.plot([], [], 'r.')
    plt.xlim(-1, -0.3)
    plt.ylim(0.1, 2)
    plt.xlabel('-Coverage')
    plt.ylabel('Flight Time')
    plt.title('Pareto History')
    animate = animation.FuncAnimation(fig1, update_pareto, num_gen,
                            fargs=(data, l), interval=50, blit=False)
    fig1.show(animate)
    return animate
    # this save file is kind of a bug, but it lets you get the images before it
    # finishes at high quality.
    # animate.save('frames.mp4',
    #                        writer = 'writer',codec = 'mp4')

def plotFitness(pareto_hist):
    pareto_hist = np.array(pareto_hist)
    fig2= plt.figure()
    cov = []
    ft = []
    for gen in pareto_hist:
        cov.append(np.mean(gen[:,0]))
        ft.append(np.mean(gen[:,1]))
    num_gens = pareto_hist.shape[0]
    t = range(num_gens)
    data1 = cov
    data2 = ft

    fig2, ax1 = plt.subplots()

    color = 'black'
    ax1.set_xlabel('Generations')
    ax1.set_ylabel('-Coverage', color=color)
    ax1.plot(t, data1, color=color, label="Coverage")
    ax1.tick_params(axis='y', labelcolor=color)

    # instantiate a second axes that shares the same x-axis
    ax2 = ax1.twinx()

    color = 'blue'
    # we already handled the x-label with ax1
    ax2.set_ylabel('Flight Time', color=color)
    ax2.plot(t, data2, color=color, label="Flight Time")
    ax2.tick_params(axis='y', labelcolor=color)
    # otherwise the right y-label is slightly clipped
    fig2.tight_layout()
    fig2.show()
# ==============================================================================
# ==============================================================================

inv_360 = 1 / 360
inv_2pi = 0.5 / np.pi

def deg_wrap_180( angle ):
    angle -= 360.0 * np.floor((angle + 180.) * inv_360)
    return angle

def deg_wrap_360( angle ):
    angle -= 360.0 * np.floor(angle * inv_360)
    return angle

def rad_wrap_pi( angle ):
    angle -= 2*np.pi * np.floor((angle + np.pi) * inv_2pi)
    return angle

def rad_wrap_2pi( angle ):
    angle -= 2*np.pi * np.floor(angle * inv_2pi)
    return angle

# ==============================================================================
# ==============================================================================

def getRot2D(theta):
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta),  np.cos(theta)]])

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

    # define points on the outside of the view
    for angle in ray_angles:
        rot = getRot2D(angle)
        pts.append(rot.dot([0, max_view/scale]))

    return np.array(pts)

# ==============================================================================
# ==============================================================================

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

    for m in range(M):
        u = r + m/M
        while u > c:
            i += 1
            c = c + w[i]

        new_x = copy.deepcopy(X[i])
        Xbar.append(new_x)

    return Xbar


def sortBy(stuff, stuff_values):
    order = np.argsort(stuff_values)
    stuff = list(np.array(stuff)[order])
    stuff_values = list(np.array(stuff_values)[order])
    return stuff, stuff_values

# ======================================
# ======================================

def giveMeOptions(graph, pt1, pt2):
    options = np.where(graph[pt1]*graph[pt2])
    return options

# ======================================
# ======================================

def getObjValsList( population ):
    objs = [thing._obj_val for thing in population._gen_parent]
    return objs

def savePopulation(population,filename):
    pickle.dump(population, open(filename, "wb"))

def loadPopulation(filename):
    population = pickle.load( open(filename, "rb"))
    return population

# =======================================
# =======================================


def findNearest(array, value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return array[idx-1]
    else:
        return array[idx]


def getCandidates(num_agents, path, window_poss, offset_window, offset_threshold):
    offset_candidates = np.zeros((num_agents, num_agents, 2*offset_window))
    feasibility = np.ones((num_agents, num_agents, 2*offset_window))
    for agent1 in range(num_agents):
        for agent2 in range(agent1+1, num_agents):
            path1 = path[agent1][path[agent1] != -1]
            path2 = path[agent2][path[agent2] != -1]
            brd_wps = np.abs(path1[:, None] -
                             path2[None, :])
            common_wps = np.array(np.where(brd_wps == 0)).T
            wp_diffs = np.diff(common_wps)
            sorted_diffs = np.sort(wp_diffs.T)
            for ii, offset in enumerate(window_poss):
                if ((sorted_diffs - offset) == 0).any():
                    feasibility[agent1, agent2, ii] = 0

                offset_candidates[agent1, agent2, ii] = \
                    (np.sum(abs(sorted_diffs-offset) < offset_threshold))
            if (offset_candidates[agent1,agent2] != 0).all():
                print ("No clean path between agent", agent1, "and", agent2, "try relaxing constraints")

    return feasibility, offset_candidates


def findBestOffset(num_agents, window_poss, offset_candidates, feasibility):
    offset_mat = np.zeros((num_agents, num_agents)).astype(int)
    offset_poss = np.zeros((num_agents, num_agents)).astype(object)
    for agent1 in range(num_agents):
        for agent2 in range(agent1+1, num_agents):
            try:
                best = findNearest(
                    window_poss[offset_candidates[agent1, agent2, :] == 0], 0)
                offset_poss[agent1,
                           agent2] = window_poss[offset_candidates[agent1, agent2, :] == 0]
            except:
                if (feasibility[agent1, agent2, :] == 1).any():
                    feasible_cand = offset_candidates[agent1,
                                                      agent2, :][feasibility[agent1, agent2, :] == 1]
                    feasible_pos = window_poss[feasibility[agent1,
                                                           agent2, :] == 1]
                else:
                    print("Feasible path not possible between", agent1,
                          "and", agent2, "relax constraints, or fly with caution")
                    feasible_cand = offset_candidates[agent1, agent2, :]
                    feasible_pos = window_poss
                min_cand = feasible_cand == min(feasible_cand)
                offset_poss[agent1, agent2] = feasible_pos[min_cand]
                best = findNearest(feasible_pos[min_cand], 0)
                    
            offset_mat[agent1, agent2] = best
    return offset_mat, offset_poss


def getPathOffsets(path, offset_window, offset_threshold):
    num_agents = path.shape[0]
    window_poss = np.arange(-offset_window, offset_window, 1)
    feasibility, offset_candidates = getCandidates(num_agents, path, window_poss, offset_window,offset_threshold)
    offset_mat, offset_poss = findBestOffset(num_agents, window_poss,offset_candidates,feasibility)
    for agent2 in range(2,num_agents):
        for agent1 in range(num_agents):
            offset_poss[agent1, agent2] -= offset_mat[agent1, agent2-1]

        offsets = offset_poss[0:agent2,agent2]
        new_cand = reduce(np.intersect1d,offsets)
        try:
            best = findNearest(new_cand, 0)
            for agent1 in range(agent2):
                offset_mat[agent1,agent2] = best + offset_mat[agent1,agent2-1]
        except:
            print("Could not find feasible path offset for agent", agent1,"relax constraint or fly with caution")
    return offset_mat[0,:]


def splitLoopClosures(lc):
    lcs = []
    for ii in range(len(lc)):
        for jj in range(ii+1,len(lc)):
            lcs.append([lc[ii],lc[jj]])
    return np.array(lcs)


def formatLoopClosures(path,loop_closures,path_splits):
    num_agents = path.shape[0]
    lc_array = np.array([[0,0]])
    for lc in loop_closures:
        if len(lc)>2:
            lcs = splitLoopClosures(lc)
            lc_array = np.vstack((lc_array,lcs))
        elif len(lc)==2:
            lc_array = np.vstack((lc_array,lc))
    lc_array = np.delete(lc_array,0,0)
    lc_array = np.sort(lc_array)
    agent_num = np.digitize(lc_array,path_splits)
    forecasted_lc = np.stack((agent_num,lc_array))
    path_splits_idx = np.insert(path_splits,0,0)
    for ii in range(num_agents):
        forecasted_lc[1][agent_num == ii] -= path_splits_idx[ii]
    return forecasted_lc


def pruneWps(wps):
    prev_angle = None
    pruned_pts = []
    wp_idx = np.arange(0,len(wps),1)
    for ii,wp in enumerate(wps):
        if wp[2]==prev_angle:
            pruned_pts.append(ii)
        prev_angle = wp[2]
    pruned_pts = np.array(pruned_pts)
    pruned_pts = pruned_pts[pruned_pts!=len(wps)]
    wp_idx = np.delete(wp_idx,pruned_pts)
    wps = np.delete(wps,pruned_pts,0)
    return wps, wp_idx

def pruneLoopClosures(unpruned_lcs,wp_idx,num_agents):
    lcs = unpruned_lcs.copy()
    for agent in range(num_agents):
        lcs[1][lcs[0] == agent] = np.digitize(
            lcs[1][lcs[0] == agent], wp_idx[agent])
    lcs = np.unique(lcs, axis=1)
    return lcs

def generateWaypointData(organism, height, offset_window, offset_threshold, prune=True):
    path = organism._dna
    wp_locs = organism._pather._XY*organism._scale
    trav_angs = organism._mappy._traverse_angles
    num_agents = path.shape[0]
    unpruned_waypoints = []
    waypoints = []
    wp_idx = []
    for agent in range(num_agents):
        current_path = path[agent][path[agent] !=-1]
        waypoint = wp_locs[current_path]
        path_diff = np.vstack((current_path,np.roll(current_path,-1))).T
        angle = np.expand_dims(trav_angs[path_diff[:,0],path_diff[:,1]],1)
        wp_height = np.expand_dims(np.ones(len(current_path))*height,1)
        unpruned_wps = np.hstack((waypoint, angle, wp_height))
        pruned_wps, pruned_wp_idx = pruneWps(unpruned_wps)
        waypoints.append(pruned_wps)
        unpruned_waypoints.append(unpruned_wps)
        wp_idx.append(pruned_wp_idx)
    path_offset = getPathOffsets(path, offset_window, offset_threshold)
    loop_closures, path_splits = organism._mappy.getLoopClosures(
        path, return_lc_path = True)
    unpruned_lcs = formatLoopClosures(path,loop_closures, path_splits)
    lcs = pruneLoopClosures(unpruned_lcs,wp_idx,num_agents)

    if prune == True:
        return waypoints, lcs, path_offset
    else: 
        return unpruned_waypoints, unpruned_lcs, path_offset


def savePath(organism, height, offset_window, offset_threshold, prune=False):
    waypoints, lcs, path_offset = generateWaypointData(
        organism, height, offset_window, offset_threshold, prune)
    np.savez("waypoints.npz", *waypoints[0:], lcs=lcs.astype(np.double), offsets=path_offset.astype(np.double))
            
