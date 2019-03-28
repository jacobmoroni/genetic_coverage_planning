from IPython.core.debugger import set_trace
from importlib import reload

import numpy as np

# TODO: import classes to compute both objective values
# TODO: import fitness class (operates on both objectives)
import pathmaker
# import my_module as mym
reload(pathmaker)
# reload(objective)
# from objective import Objective
from pathmaker import PathMaker

class GeneticalGorithm( ):
    def __init__( self, mappy, scale, narrowest_hall, max_dna_len, pather ):
        # setup params
        self._G_sz = 10 # has to be an EVEN number !!!!!
        self._G_num = 10
        self._T_sz = 4
        self._gamma = 0.5
        self._xover_probability = 0.5
        self._xover_param = 0.5
        self._mutate_probability = 0.5
        self._mutate_param = 0.5

        # list for holding all chromosomes in parent generation
        self._gen_parent = []
        self._gen_parent_fit = []
        # list for holding all chromosomes in children generation
        self._gen_child = []
        self._gen_child_fit = []

        self._gen_pc = []
        self._gen_pc_fit = []

        # create list of first generation
        self.firstGeneration(mappy, scale, narrowest_hall, max_dna_len, pather)

        # start species history
        for ii in range(self._G_num):
            for jj in range(self._G_sz * 0.5):

        #
    #
    def firstGeneration(self, mappy, scale, narrowest_hall, max_dna_len, pather):
        # create random set of chromosomes
        path_length = 200
        start_idx = 207
        # pool this for loop with multiprocess
        for ii in len(self._G_sz):
            rand_path = pather.makeMeAPath(path_length,start_idx)
            self._gen_parent.append( Organism(rand_path, mappy, scale, narrowest_hall, max_dna_len, pather) )
            self._gen_parent_fit.append(self._gen_parent[ii]._fitness)
        #
        self.maximin()
        # TODO sort according to fitness function

    #
    def selection(self):
        # tourny or [ roulette ]
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
        # TODO: obj1 is coverage --> MINIMIZING the NEGATIVE
        # obj2 is flight-time --> MINIMIZING
        # obj1 = -self._fitness[0]
        # obj1 = self._fitness[1]

        comp_min = []
        comp_max = []

        gen_pc_fit = []
        gen_pc_fit.append(self._gen_parent_fit)
        gen_pc_fit.append(self._gen_child_fit)

        for ii, gen_fit_1 in enumerate(gen_pc_fit):
            for jj, gen_fit_2 in enumerate(gen_pc_fit):
                if ii == jj:
                    continue
                else:
                    comp_min.append( np.min([gen_fit_1[0]-gen_fit_2[0], gen_fit_1[1]-gen_fit_2[1]]) )
                #
            #
            comp_max.append( np.max(comp_min) )
        #
        sort_idx = np.argsort(comp_max)
        gen_pc = np.append
        self._gen_pc.append(self._gen_parent)
        self._gen_pc.append(self._gen_child)

        self._gen_pc_fit.append(self._gen_parent)
        self._gen_pc_fit.append(self._gen_child)
    #
#

class Organism( ):
    def __init__(self, dna, mappy, scale, narrowest_hall, max_dna_len, pather):
        # [ value ] vs binary
        # self.num_genes = 50
        # self.dna = []

        # self._dna = path
        # self._dna = []
        # self._mappy = mym.mappy
        self._mappy = mappy
        self._scale = scale
        self._narrowest_hall = narrowest_hall
        self._safety_buffer = narrowest_hall * 0.5
        self._max_dna_len = max_dna_len
        self._pather = pather

        len_dna = len(dna)

        self._len_dna = len_dna
        self._dna = dna

        # if len_dna > self._max_dna_len - 1:
        #     self._dna = self._dna[0:self._max_dna_len]
        # elif len_dna < self._max_dna_len - 1:
        #     self._dna = np.append( self._dna, np.ones(max_dna_len - len_dna - 1) * -1 ).astype(int)
        # #
        self.padDna()


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

        # apply mutation
        self.mutation()

        self._obj_val = self.calcObj()
        # self._obj_val_2 = self.calcObj2()

        self._fitness = self.maximin()
    #
    def calcObj(self):
        coverage = self._mappy.getCoverage(self._dna)
        return [self._dna[0], coverage]
    #

    def crossover(self, mate):
        # TODO: remove static probability
        x_prob = 0.2 # np.random.rand()
        if x_prob < 0.5:
            xover_pts = self.matchWaypt(mate._dna, mate._len_dna)
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

        lil_timmy = Organism(dna1, self._mappy, self._scale, self._narrowest_hall, self._max_dna_len, self._pather)
        lil_susy = Organism(dna2, self._mappy, self._scale, self._narrowest_hall, self._max_dna_len, self._pather)
        return lil_timmy, lil_susy
    #
    def mutation(self):
        # uniform vs [ dynamic ]
        # TODO: remove static probability
        x_prob = 0.2 # np.random.rand()
        if x_prob < 0.5:
            idx = np.random.randint(0,self._len_dna)
            len_tail = self._len_dna - idx
            dna_tail = self._pather.makeMeAPath(len_tail, self._dna[idx])
            dna_head = self._dna[0:idx]
            self._dna = np.append(dna_head, dna_tail)
        #
        self.padDna()

    #
    def calcConstrS(self):
        # return array of all constraint eq vals
        # maybe check here instead which constraints are feasible
        pass
    #
    def matchWaypt(self, dna_mate, len_dna_mate):
        # set_trace()
        xover_pts = []
        # broadcast is brd
        brd_diff_mat = np.abs(self._dna[1:self._len_dna+1,None] - dna_mate[None,1:len_dna_mate+1])

        un_pruned_pts = np.array(np.where(brd_diff_mat == 0)).T

        xover_tf = np.abs( un_pruned_pts[:,0] - un_pruned_pts[:,1] ) < self._time_thresh
        xover_pts = un_pruned_pts[xover_tf]

        return xover_pts
    #
    def padDna(self):
        if self._len_dna > self._max_dna_len:
            self._dna = self._dna[0:self._max_dna_len]
        elif self._len_dna < self._max_dna_len:
            self._dna = np.append( self._dna, np.ones(self._max_dna_len - self._len_dna) * -1 ).astype(int)
        #
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
