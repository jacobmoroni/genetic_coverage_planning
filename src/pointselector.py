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
# plt.ion()

import gori_tools as got
reload(got)

class PointSelector(object):
    def __init__(self,points,objs,mappy,pather,population,fig):

        self.mappy = mappy
        self.pather = pather
        self.population = population
        self._x = 0
        self._y = 0
        self._points = points
        self._objectives = objs
        self._fig = fig
        self._cid = points.figure.canvas.mpl_connect('button_press_event', self)
        nearest_point = -1

    def __call__(self, event):

        # Records x and y locations of mouse clicks and sends them to start and goal positions
        cv2.destroyAllWindows()
        if event.inaxes!=self._points.axes: return

        self._x = event.xdata
        self._y = event.ydata

        obj_diff = self._objectives - np.array([self._x,self._y])
        nearest_point = np.argmin(np.linalg.norm(obj_diff,axis=1))
        # print (nearest_point)
        self._points.set_data(self._objectives[nearest_point,0],self._objectives[nearest_point,1])
        self._points.figure.canvas.draw()
        current_organism = self.population._gen_parent[nearest_point]
        # self.mappy.visualizePath(self.pather._XY,current_organism._dna[0:current_organism._len_dna],self._fig)
        coverage, travel_dist, coverage_map = self.mappy.getCoverage(current_organism._dna,return_map=True)
        num_lcs,loop_closures = self.mappy.getSoloLoopClosures(current_organism._dna, return_loop_close=True)
        num_lcs,combo_closures = self.mappy.getCombLoopClosures(current_organism._dna)
        self.mappy.visualizePathWithCoverage(self.pather._XY,
                                             current_organism._dna,
                                             self._fig,
                                             coverage_map,
                                             loop_closures,
                                             combo_closures,
                                             coverage,
                                             travel_dist,
                                             nearest_point)

        return [self._x,self._y]
