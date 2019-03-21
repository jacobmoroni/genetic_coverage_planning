from IPython.core.debugger import set_trace
from importlib import reload

import numpy as np

# TODO: import classes to compute both objective values
# TODO: import fitness class (operates on both objectives)
import objective
reload(objective)
from objective import Objective

class GeneticAlgorithm( ):
    def __init__( self ):
        pass
    #
    def new_generation(self):
        pass
    #
    def selection(self):
        pass
    #
    def crossover(self):
        pass
    #
    def mutation(self):
        pass
    #
    def elitism(self):
        pass
    #
    def fitness(self):
        # combines both objectives with maximin into fitness
        pass
    #

#

#
