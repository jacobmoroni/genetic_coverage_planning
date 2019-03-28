from IPython.core.debugger import set_trace
from importlib import reload

import numpy as np
import copy

# TODO: import classes to compute both objective values
# TODO: import rackNstack class (operates on both objectives)
import pathmaker
# import my_module as mym
reload(pathmaker)
# reload(objective)
# from objective import Objective
from pathmaker import PathMaker
from tqdm import tqdm

def lowVarSample(X, fitnesses, pressure):
    # pressure should be between 0 and 1
    log_w = -np.array(fitnesses)
    log_w = log_w - np.max(log_w)
    base = pressure + 1
    w = base**log_w
    w = w/np.sum(w)
    Xbar = []
    M = len(X)
    r = np.random.uniform(0, 1/M)
    c = w[0]
    i = 0
    last_i = i
    for m in range(M):
        u = r + m/M
        while u > c:
            i += 1
            c = c + w[i]
        new_x = copy.deepcopy(X[i])
        Xbar.append(new_x)
        last_i = i
    return Xbar

def sortBy(stuff, stuff_values):
    order = np.argsort(stuff_values)
    stuff = list(np.array(stuff)[order])
    stuff_values = list(np.array(stuff_values)[order])
    return stuff, stuff_values


class GeneticalGorithm( ):
    def __init__( self, mappy, scale, narrowest_hall, max_dna_len, pather ):
        # setup params
        self._G_sz = 100 # has to be an EVEN number !!!!!
        # self._G_num = 10
        self._tourney_sz = 4
        self._gamma = 0.5 # roulette exponent >= 0. 0 means zero fitness pressure


        # list for holding all chromosomes in parent generation
        self._gen_parent = []
        self._gen_parent_fit = []

        # self._gen_pc = []
        # self._gen_pc_fit = []
        # self._gen_parent_fit_rel = None
        # self._gen_child_fit_rel = None

        # create list of first generation
        self.firstGeneration(mappy, scale, narrowest_hall, max_dna_len, pather)

        #
    #
    def firstGeneration(self, mappy, scale, narrowest_hall, max_dna_len, pather):
        # create random set of chromosomes
        path_length = 200
        start_idx = 207
        # pool this for loop with multiprocess
        for ii in range(self._G_sz):
            rand_path = pather.makeMeAPath(path_length,start_idx)
            self._gen_parent.append( Organism(rand_path, mappy, scale, narrowest_hall, max_dna_len, pather) )
        #
        # map is a python built-in
        self._gen_parent_fit = self.rackNstack(self._gen_parent)
        #
        # self._gen_parent = np.array(self._gen_parent)
        # self._gen_parent_fit = np.array(self._gen_parent_fit)
    #
    def runEvolution(self, num_generations):
        # list for holding all chromosomes in children generation
        for ii in tqdm(range(num_generations), desc="Evolving"):
            gen_child = []
            gen_child_fit = []
            # grab a set of parents that are good to reproduce
            strong_parents = lowVarSample(self._gen_parent, self._gen_parent_fit, self._gamma)
            for mommy, daddy in zip(strong_parents[:self._G_sz//2], strong_parents[self._G_sz//2:]):
                # do crossover with app the worthy parents
                family_kids = mommy.crossover(daddy)
                # family_kids_fit = map(self.rackNstack, family_kids)#[kid.rackNstack for kid in family_kids]
                gen_child += family_kids
                # gen_child_fit += family_kids_fit
            #
            # only the strong will survive this arena
            # it's dog eat dog, parent eat child. Animals.
            self.arena(self._gen_parent + gen_child)
            #
        #
    #

    #

    def arena(self, gladiators):
        #rack and stack
        gladiators_fit = self.rackNstack(gladiators)
        gladiators, gladiators_fit = sortBy(gladiators, gladiators_fit)
        #kill the unfit ones
        self._gen_parent = gladiators[:self._G_sz]
        self._gen_parent_fit = gladiators_fit[:self._G_sz]
    #
    # scoring vs ranking vs [ maximin ]
    def rackNstack(self, gladiators):
        # combines both objectives with maximin into rackNstack
        # penalty vs [ segregation ]

        # TODO: obj1 is coverage --> MINIMIZING the NEGATIVE
        # obj2 is flight-time --> MINIMIZING
        # obj1 = -self._rackNstack[0]
        # obj1 = self._rackNstack[1]

        comp_min = []
        comp_max = []

        # gen_pc_fit = []
        # gen_pc_fit.extend(self._gen_parent_fit)
        # gen_pc_fit.extend(self._gen_child_fit)

        for ii, gen_fit_1 in enumerate(gladiators):
            comp_min = []
            for jj, gen_fit_2 in enumerate(gladiators):
                if ii == jj:
                    continue
                else:
                    comp_min.append( np.min([gen_fit_1._obj_val[0]-gen_fit_2._obj_val[0], gen_fit_1._obj_val[1]-gen_fit_2._obj_val[1]]) )
                #
            #
            comp_max.append( np.max(comp_min) )
        #
        # sort_idx = np.argsort(comp_max)[:self._G_sz]
        #
        # # self._gen_parent = np.array(gladiators)[sort_idx]
        # self._gen_parent_fit = np.array(gen_pc_fit)[sort_idx]

        fitnesses = comp_max
        return fitnesses
    #
    def pop_sort(self, gen_list, fit_list): # sort population
        pass
    #
#

class Organism( ):
    def __init__(self, dna, mappy, scale, narrowest_hall, max_dna_len, pather):
        # [ value ] vs binary
        # self.num_genes = 50
        # self.dna = []
        self._xover_probability = 0.5
        # self._xover_param = 0.5
        self._mutate_probability = 0.5
        # self._mutate_param = 0.5
        # self._obj_scale

        # self._dna = path
        # self._dna = []
        # self._mappy = mym.mappy
        self._mappy = mappy
        self._scale = scale
        self._narrowest_hall = narrowest_hall
        self._safety_buffer = narrowest_hall * 0.5
        self._max_dna_len = max_dna_len
        self._pather = pather
        self._ft_scale  = 0.0001

        len_dna = len(dna)

        self._len_dna = len_dna
        self._dna = dna

        # if len_dna > self._max_dna_len - 1:
        #     self._dna = self._dna[0:self._max_dna_len]
        # elif len_dna < self._max_dna_len - 1:
        #     self._dna = np.append( self._dna, np.ones(max_dna_len - len_dna - 1) * -1 ).astype(int)
        # #

        self.addTelomere()

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


    #
    def calcObj(self):
        coverage, travel_dist = self._mappy.getCoverage(self._dna)
        travel_dist = travel_dist * self._ft_scale
        return [coverage, travel_dist]
    #

    def crossover(self, mate):
        # TODO: remove static probability
        uniform_num = np.random.rand()
        xover_pts = self.matchWaypt(mate._dna, mate._len_dna)
        if uniform_num < self._xover_probability and len(xover_pts) > 0:

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

        # self.addTelomere(dna1)
        # self.addTelomere(dna2)

        lil_timmy = Organism(dna1[:self._max_dna_len+1], self._mappy, self._scale, self._narrowest_hall, self._max_dna_len, self._pather)
        lil_susy = Organism(dna2[:self._max_dna_len+1], self._mappy, self._scale, self._narrowest_hall, self._max_dna_len, self._pather)
        return [lil_timmy, lil_susy]
    #
    def mutation(self):
        # uniform vs [ dynamic ]
        # TODO: remove static probability
        uniform_num = np.random.rand()
        if uniform_num < self._mutate_probability:
            idx = np.random.randint(0,self._len_dna)
            len_tail = self._len_dna - idx
            try:
                dna_tail = self._pather.makeMeAPath(len_tail, self._dna[idx])
            except:
                set_trace()
            dna_head = self._dna[0:idx]
            self._dna = np.append(dna_head, dna_tail)
        #
        self.addTelomere()

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
    def addTelomere(self):
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
#     def calc_rackNstack( self ):
#         pass
#     #
# #

#
