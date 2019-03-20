#! /usr/bin/env python3

from IPython.core.debugger import set_trace
from importlib import reload

import numpy as np
import cv2

import pathmaker
reload(pathmaker)
from pathmaker import PathMaker

file_name = "map_scaled.png"
scale = 0.15
narrowest_hall = 1.75
safety_buffer = 0.7
map_scaled = cv2.imread(file_name,cv2.IMREAD_GRAYSCALE)

pather = PathMaker(map_scaled, scale, narrowest_hall, safety_buffer)

pather.smartly_place_dots()
