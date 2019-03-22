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
        # setup params
        self._G_sz = 10
        self._G_num = 10
        self._T_sz = 4
        self._gamma = 0.5
        self._xover_probability = 0.5
        self._xover_param = 0.5
        self._mutate_probability = 0.5
        self._mutate_param = 0.5

        # list for holding all chromosomes in parent generation
        self._gen_parent = []
        # list for holding all chromosomes in children generation
        self._gen_child = []
    #
    def first_generation(self):
        # create random set of chromosomes
        for ii in len(self._G_sz):
            rand_path = gen_path()
            self._gen_parent.append( Chromosome(rand_path) )
        #
    #
    def selection(self):
        # tourny or [ roulette ]
        pass
    #
    # def crossover(self):
    #     # single-point vs uniform vs [ blend ]
    #     pass
    # #
    def mutation(self):
        # uniform vs [ dynamic ]
        pass
    #
    def elitism(self):
        pass
    #
    def fitness(self):
        # combines both objectives with maximin into fitness
        # penalty vs [ segregation ]
        pass
    #
    # scoring vs ranking vs [ maximin ]
    def maximin(self):
        pass
    #
#

class Chromosome( ):
    def __init__(self, path, mappy, scale, hall_width, safety_buffer):
        # [ value ] vs binary
        # self.num_genes = 50
        # self.dna = []
        # for ii in len( self.num_genes ):
        #     self.dna.append( path[ii] )
        # #
        self._dna = path
        self._mappy = mappy
        self._scale = scale
        self._hall_width = hall_width
        self._safety_buffer = hall_width/2#safety_buffer

        # comput values of both objectives
        self._obj_vals = calc_obj(map)

        # compute constraints for designs, check feasibility
        self._constr_1 = 1
        self._constr_feas = []
        self._constr_infeas = []
        # for ii in len( num_constr ):
        #     constr_ii = 1
        #     self.constr_s.append( )
        # #
        self._constr_vals = calc_constr_s()

        self.xover_max_t_sep = None
    #
    def calc_obj(self):
        coverage = self._mappy.getCoverage(self._dna)
        return [self._dna[0], coverage]
    #

    def crossover(self, mate):
        dna1 = []
        dna2 = []

        x_prob = np.random.rand()
        if x_prob < 0.5:
            dna1.append(jj)
            dna2.append(ii)
        else:
            dna1.append(ii)
            dna2.append(jj)
        #

        lil_timmy = Chromosome(dna1)
        lil_susy = Chromosome(dna2)
        return lil_timmy, lil_susy
    #


    def calc_constr_s(self):
        # return array of all constraint eq vals
        # maybe check here instead which constraints are feasible
        pass
    #
    def match_waypt(self, dna_mate):
        # wpt is waypoint
        for each forckade in self.dna:
            ai = np.where(dna_mate.dna == forckade)
    #
#

# class Generation( ):
#     def __init__( self ):
#         pass
#     #
#     def calc_fitness( self ):
#         pass
#     #
# #

#
