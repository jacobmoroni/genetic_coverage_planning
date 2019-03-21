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
        self.G_sz = 10
        self.G_num = 10
        self.T_sz = 4
        self.gamma = 0.5
        self.xover_probability = 0.5
        self.xover_param = 0.5
        self.mutate_probability = 0.5
        self.mutate_param = 0.5

        # list for holding all chromosomes in parent generation
        self.gen_parent = []
        # list for holding all chromosomes in children generation
        self.gen_child = []
    #
    def first_generation(self):
        # create random set of chromosomes
        for ii in len(self.G_sz):
            rand_path = gen_path()
            self.gen_parent.append( Chromosome(rand_path) )
        #
    #
    def selection(self):
        # tourny or [ roulette ]
        pass
    #
    def crossover(self):
        # single-point vs uniform vs [ blend ]
        pass
    #
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
    def __init__(self, path):
        # [ value ] vs binary
        # self.num_genes = 50
        # self.dna = []
        # for ii in len( self.num_genes ):
        #     self.dna.append( path[ii] )
        # #
        self.dna = path

        # comput values of both objectives
        self.obj_value_1 = calc_obj_1( )
        self.obj_value_2 = calc_obj_2( )

        # compute constraints for designs, check feasibility
        self.constr_1 = 1
        self.constr_feas = []
        self.constr_infeas = []
        # for ii in len( num_constr ):
        #     constr_ii = 1
        #     self.constr_s.append( )
        # #
        self.constr_vals = calc_constr_s( )

    #
    def calc_obj_1(self):
        pass
    #
    def calc_obj_2(self):
        pass
    #
    def calc_constr_s(self):
        # return array of all constraint eq vals
        # maybe check here instead which constraints are feasible
        pass
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
