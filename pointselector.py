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
from matplotlib import pyplot as plt
# plt.ion()


class PointSelector(object):
    def __init__(self,points,objs,mappy,pather,population):
        
        self.mappy = mappy
        self.pather = pather
        self.population = population
        self._x = 0
        self._y = 0
        self._points = points
        self._objectives = objs
        self._cid = points.figure.canvas.mpl_connect('button_press_event', self)
        nearest_point = -1
   
    def __call__(self, event):
        
        #Records x and y locations of mouse clicks and sends them to start and goal positions
        cv2.destroyAllWindows()
        if event.inaxes!=self._points.axes: return
        
        self._x = event.xdata
        self._y = event.ydata

        obj_diff = self._objectives - np.array([self._x,self._y])
        nearest_point = np.argmin(np.linalg.norm(obj_diff,axis=1))

        self._points.set_data(self._objectives[nearest_point,0],self._objectives[nearest_point,1])
        self._points.figure.canvas.draw()
        current_organism = self.population._gen_parent[nearest_point]
        self.mappy.visualizePath(self.pather._XY,current_organism._dna[0:current_organism._len_dna])
        # self.show_path(mappy,population[nearest_point])
        return [self._x,self._y]
    # def show_path(nearest_point, mappy, 

def main():

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('Select Start and Goal (Close when finished)')

    line, = ax.plot([0,0], [0,0],"xr")  # empty line
    pointy = PointSelector(line)
    plt.plot(0,0,'.r')
    plt.show()
    print (plotty.x)

if __name__ == '__main__':
    main()
