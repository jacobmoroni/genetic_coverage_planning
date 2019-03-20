from IPython.core.debugger import set_trace
from importlib import reload

import numpy as np
import cv2
import matplotlib.pyplot as plt

class PathMaker(object):
    def __init__(self, mappy, scale, hall_width):
        self.mappy = mappy
        self.scale = scale
        self.hall_width = hall_width
    #
    def smartly_place_dots(self):
        self.dots = np.zeros_like(self.mappy)
        stride = int(self.hall_width//self.scale)
        print(self.dots.shape)
        for i in 
        self.dots[0:stride:-1, 0:stride:-1] = 255
        cv2.imshow('dots',self.dots)
        cv2.waitKey()
    #
#
