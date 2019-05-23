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
import cv2.cv2 as cv2
from matplotlib import pyplot as plt

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

    def __call__(self, event):

        # Records x and y locations of mouse clicks and sends them to start and goal positions
        cv2.destroyAllWindows()
        if event.inaxes!=self._points.axes: return

        self._x = event.xdata
        self._y = event.ydata

        obj_diff = self._objectives - np.array([self._x,self._y])
        nearest_point = np.argmin(np.linalg.norm(obj_diff,axis=1))
        self._points.set_data(self._objectives[nearest_point,0],self._objectives[nearest_point,1])
        self._points.figure.canvas.draw()
        current_organism = self.population._gen_parent[nearest_point]
        coverage, travel_dist, coverage_map = self.mappy.getCoverage(current_organism._dna,return_map=True)
        _,loop_closures = self.mappy.getLoopClosures(current_organism._dna, return_loop_close=True)
        if event.button == 1:
            self.mappy.visualizePathWithCoverage(self.pather._XY,
                                             current_organism._dna,
                                             self._fig,
                                             coverage_map,
                                             loop_closures,
                                             coverage,
                                             travel_dist,
                                             nearest_point)
        if event.button == 3:
            got.animateFlight(current_organism, self.pather._XY, coverage_map, self.mappy._safety_img.copy())

        return [self._x,self._y]
