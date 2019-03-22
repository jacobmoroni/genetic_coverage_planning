from IPython.core.debugger import set_trace
from importlib import reload

import numpy as np

# TODO: import classes to compute both objective values
# TODO: import fitness class (operates on both objectives)
# import objective
# import my_module as mym
# reload(mym)
# reload(objective)
# from objective import Objective


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
    def __init__(self, strand, mappy, scale, narrowest_hall, max_chromo_len):
        # [ value ] vs binary
        # self.num_genes = 50
        # self.dna = []

        # self._dna = path
        # self._dna = []
        # self._mappy = mym.mappy
        self._mappy = mappy
        self._scale = scale
        self._hall_width = narrowest_hall
        self._safety_buffer = narrowest_hall * 0.5
        self._max_chromo_len = max_chromo_len

        len_strand = len(strand)

        self._dna = strand

        if len_strand > self._max_chromo_len - 1:
            self._dna = self._dna[0:self._max_chromo_len]
        elif len_strand < self._max_chromo_len - 1:
            self._dna = np.append( self._dna, np.ones(max_chromo_len - len_strand - 1) * -1 ).astype(int)
        #
        self._dna = np.append( len_strand, self._dna)


        # comput values of both objectives
        # self._obj_vals = self.calc_obj(map)
        # self._obj_vals = self.calc_obj()

        # compute constraints for designs, check feasibility
        self._constr_1 = 1
        self._constr_feas = []
        self._constr_infeas = []
        # for ii in len( num_constr ):
        #     constr_ii = 1
        #     self.constr_s.append( )
        # #
        # self._constr_vals = calc_constr_s()

        # self.xover_max_t_sep = None
        self._time_thresh = 70
    #
    def calc_obj(self):
        coverage = self._mappy.getCoverage(self._dna)
        return [self._dna[0], coverage]
    #

    def crossover(self, mate):

        x_prob = 0.2 # np.random.rand()
        if x_prob < 0.5:
            xover_pts = self.match_waypt(mate._dna)
            xover_idx = np.arange( len(xover_pts) )
            single_idx = np.random.choice( xover_idx )
            single_pt = xover_pts[single_idx]
            dna1 = self._dna[1:single_pt[0]]
            dna2 = mate._dna[1:single_pt[1]]
            dna1 = np.append( dna1, mate._dna[single_pt[1]:])
            dna2 = np.append( dna2, self._dna[single_pt[0]:])
        else:
            dna1 = self._dna
            dna2 = mate._dna
        #
        idx = np.where(dna1 == -1)[0]
        dna1 = np.delete(dna1,idx)
        idx = np.where(dna2 == -1)[0]
        dna2 = np.delete(dna2,idx)

        lil_timmy = Chromosome(dna1, self._mappy, self._scale, self._narrowest_hall, self._max_chromo_len)
        lil_susy = Chromosome(dna2, self._mappy, self._scale, self._narrowest_hall, self._max_chromo_len)
        return lil_timmy, lil_susy
    #


    def calc_constr_s(self):
        # return array of all constraint eq vals
        # maybe check here instead which constraints are feasible
        pass
    #
    def match_waypt(self, dna_mate):

        xover_pts = []
        # broadcast is brd
        brd_mat = np.abs(self._dna[2:self._dna[0],None] - dna_mate[None,2:dna_mate[0]])

        un_pruned_pts = np.array(np.where(brd_mat == 0)).T

        xover_tf = np.abs( un_pruned_pts[:,0] - un_pruned_pts[:,1] ) < self._time_thresh
        xover_pts = un_pruned_pts[xover_tf]

        return xover_pts
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
