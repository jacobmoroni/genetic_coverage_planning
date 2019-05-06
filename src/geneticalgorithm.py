from IPython.core.debugger import set_trace
from importlib import reload

import numpy as np
import copy
from tqdm import tqdm

import gori_tools as got
import pathmaker
reload(got)
reload(pathmaker)
from pathmaker import PathMaker


class GeneticalGorithm( ):
    def __init__( self, mappy, scale, narrowest_hall, max_dna_len, pather):
        # setup params
        self._G_sz = 100 # has to be an EVEN number !!!!!
        self._starting_path_len = 200
        # self._G_num = 10
        self._tourney_sz = 4
        self._gamma = 0.5 # roulette exponent >= 0. 0 means zero fitness pressure
        self._gen_num = 0
        # list for holding all chromosomes in parent generation
        self._gen_parent = []
        self._gen_parent_fit = []

        self._cov_constr_0 = -0.3
        self._cov_constr_f = -0.8
        self._cov_aging = 60 #how many generations it takes to get to max const
        # create list of first generation
        self.firstGeneration(mappy, scale, narrowest_hall, max_dna_len, pather)
        #
    #
    def firstGeneration(self, mappy, scale, narrowest_hall, max_dna_len, pather):
        # create random set of chromosomes
        # path_len = 100
        start_idx = 207
        # pool this for loop with multiprocess
        for ii in tqdm(range(self._G_sz), desc="Adeves (aka. Adam and Eve'ing)"):

            rand_path = pather.makeMeAPath(self._starting_path_len,start_idx)
            self._gen_parent.append( Organism(rand_path, mappy, scale, narrowest_hall, max_dna_len, pather) )
        #
        # map is a python built-in
        self._gen_parent_fit = self.rackNstack(self._gen_parent)
        #
    #
    def runEvolution(self, num_generations):
        # list for holding all chromosomes in children generation
        pareto_1_gen = []
        for ii in tqdm(range(num_generations), desc="Evolving"):
            gen_child = []
            gen_child_fit = []
            # grab a set of parents that are good to reproduce
            strong_parents = got.lowVarSample(self._gen_parent, self._gen_parent_fit, self._gamma)
            for mommy, daddy in zip(strong_parents[:self._G_sz//2], strong_parents[self._G_sz//2:]):
                # do crossover with all the worthy parents
                family_kids = mommy.crossover(daddy)
                gen_child += family_kids
            #
            # only the strong will survive this arena
            # it's dog eat dog, parent eat child. Animals.
            self.arena(self._gen_parent + gen_child)
            self._gen_num += 1

            pareto_1_gen += [got.getObjValsList(self)]
            #
        #
        return pareto_1_gen
    #
    def arena(self, gladiators):
        # rack and stack
        gladiators_fit = self.rackNstack(gladiators)
        gladiators, gladiators_fit = got.sortBy(gladiators, gladiators_fit)
        # kill the unfit ones
        self._gen_parent = gladiators[:self._G_sz]
        self._gen_parent_fit = gladiators_fit[:self._G_sz]
    #
    def rackNstack(self, gladiators):
        # combines both objectives with maximin into rackNstack
        # scoring vs ranking vs [ maximin ]
        # penalty vs [ segregation ]

        # obj1 is coverage --> MINIMIZING the NEGATIVE
        # obj2 is flight-time --> MINIMIZING
        # obj1 = -self._rackNstack[0]
        # obj2 = self._rackNstack[1]

        comp_min = []
        comp_max = []
        alpha = min(1,(self._gen_num/self._cov_aging))
        coverage_constr = (1-alpha)*self._cov_constr_0 + alpha*self._cov_constr_f
        for ii, glad in enumerate(gladiators):
            if glad._obj_val[0] > coverage_constr:
                gladiators[ii]._obj_val_sc[0] = glad._obj_val[0]
                gladiators[ii]._obj_val_sc[1] = glad._obj_val[1] + 1000.0
            else:
                gladiators[ii]._obj_val_sc[0] = glad._obj_val[0]
                gladiators[ii]._obj_val_sc[1] = glad._obj_val[1]
            #
        #

        for ii, gen_fit_1 in enumerate(gladiators):
            comp_min = []
            for jj, gen_fit_2 in enumerate(gladiators):
                if ii == jj:
                    continue
                else:
                    comp_min.append( np.min([gen_fit_1._obj_val_sc[0]-gen_fit_2._obj_val_sc[0],
                                             gen_fit_1._obj_val_sc[1]-gen_fit_2._obj_val_sc[1]]) )
                #
            #
            comp_max.append( np.max(comp_min) )
        #
        fitnesses = comp_max
        return fitnesses
    #
#

class Organism( ):
    def __init__(self, dna, mappy, scale, narrowest_hall, max_dna_len, pather):
        # TODO: I think we need to add the coverage contraint to pass into the class to make it a rolling constraint
        # [ value ] vs binary

        self._xover_probability = 0.7
        self._mutate_probability = 0.3
        self._muterpolate_probability = 0.3

        self._mappy = mappy
        self._scale = scale
        self._narrowest_hall = narrowest_hall
        self._safety_buffer = narrowest_hall * 0.5
        self._max_dna_len = max_dna_len
        self._pather = pather
        self._ft_scale  = 0.0001
        self._min_dna_len = 75
        self._num_muterpolations = 20
        self._srch_dist = 5
        self._P_muterpolate_each = 0.8
        self._min_loop_closures = 4

        len_dna = len(dna)

        self._len_dna = len_dna
        self._dna = dna


        # compute constraints for designs, check feasibility
        # incentivize straight paths
        # start with seeded design
        # min dna len
        self._constr_1 = 1
        self._constr_feas = []
        self._constr_infeas = []


        # how close the crossover points need
        # to be to eachother to be considered
        self._time_thresh = 70

        # apply mutation
        do_it = np.random.rand()
        if do_it < self._mutate_probability:
            self.mutation()
        #

        do_it = np.random.rand()
        if do_it < self._muterpolate_probability:
            self.muterpolate()
        #

        # compute values of both objectives
        self._obj_val = self.calcObj()
        self._obj_val_sc = [None,None]
        self._len_dna = len(self._dna)
        self.addTelomere()

    #
    def calcObj(self):
        coverage, travel_dist = self._mappy.getCoverageWithWalls(self._dna)
        loop_closures = self._mappy.getLoopClosures(self._dna)
        travel_dist = travel_dist * self._ft_scale
        if loop_closures < self._min_loop_closures:
            coverage = 0
        return [coverage, travel_dist]
    #
    def crossover(self, mate):
        uniform_num = np.random.rand()
        xover_pts = self.matchWaypt(mate._dna, mate._len_dna)
        # trav_pts = self.travWaypt(mate._dna, mate._len_dna)
        if uniform_num < self._xover_probability and len(xover_pts) > 0:

            xover_idx = np.arange( len(xover_pts) )
            single_idx = np.random.choice( xover_idx )

            single_pt = xover_pts[single_idx]

            dna1 = self._dna[0:single_pt[0]]
            dna2 = mate._dna[0:single_pt[1]]

            dna1 = np.append( dna1, mate._dna[single_pt[1]:])
            dna2 = np.append( dna2, self._dna[single_pt[0]:])
            for dna in [dna1,dna2]:
                if len(dna) < self._min_dna_len:
                    len_tail = self._min_dna_len - len(dna)
                    dna_tail = self._pather.makeMeAPath(len_tail, dna[-1])
                #
            #
        else:
            dna1 = self._dna
            dna2 = mate._dna
        #
        # idx = np.where(dna1 == -1)[0]
        # dna1 = np.delete(dna1,idx)
        # idx = np.where(dna2 == -1)[0]
        # dna2 = np.delete(dna2,idx)

        dna1 = dna1[dna1 != -1]
        dna2 = dna2[dna2 != -1]

        lil_timmy = Organism(dna1[:self._max_dna_len+1], self._mappy, self._scale, self._narrowest_hall, self._max_dna_len, self._pather)
        lil_susy = Organism(dna2[:self._max_dna_len+1], self._mappy, self._scale, self._narrowest_hall, self._max_dna_len, self._pather)

        return [lil_timmy, lil_susy]
    #
    def mutation(self):
        # uniform vs dynamic
        idx = np.random.randint(1,self._len_dna-1)
        len_tail = self._len_dna - idx
        try:
            if (self._len_dna+1)<self._max_dna_len:
                len_tail += np.random.randint(self._max_dna_len-self._len_dna)
        except:
            set_trace()
        dna_tail = self._pather.makeMeAPath(len_tail, self._dna[idx])
        dna_head = self._dna[0:idx]

        self._dna = np.append(dna_head, dna_tail)
        # self.addTelomere()
    #
    def muterpolate(self):

        mut_points = np.random.choice(np.arange(0, self._len_dna-(self._srch_dist+3), 1), self._num_muterpolations)
        for idx1 in mut_points:
            for idx2 in range(idx1+self._srch_dist+1, idx1+1, -1):
                do_it = np.random.rand()
                if do_it < self._P_muterpolate_each:
                    # check to see if we can get there directly
                    pt1 = self._dna[idx1]
                    if idx2 == 250:
                        set_trace()
                    pt2 = self._dna[idx2]
                    if self._pather._graph[pt1,pt2]:
                        np.delete(self._dna, np.arange(idx1+1, idx2-1, 1))
                        break
                    else:
                        # check the mutual traversible space
                        options = np.where(self._pather._graph[pt1]*self._pather._graph[pt2])[0]
                        if len(options) > 0:
                            step = np.random.choice(options)
                            self._dna[idx1+1] = step
                            np.delete(self._dna, np.arange(idx1+2, idx2-1, 1))
                            break
                        #
                    #
                #
            #
        #
        # self.addTelomere()
    #
    def calcConstrs(self):
        # return array of all constraint eq vals
        # maybe check here instead which constraints are feasible
        pass
    #
    def matchWaypt(self, dna_mate, len_dna_mate):
        keepout_idx = 1
        xover_pts = []
        # broadcast is brd
        brd_diff_mat = np.abs(self._dna[keepout_idx:self._len_dna+1,None] - dna_mate[None,keepout_idx:len_dna_mate+1])

        un_pruned_pts = np.array(np.where(brd_diff_mat == 0)).T + keepout_idx

        xover_tf = np.abs( un_pruned_pts[:,0] - un_pruned_pts[:,1] ) < self._time_thresh
        xover_pts = un_pruned_pts[xover_tf]

        return xover_pts
    #
    def travWaypt(self, dna_mate, len_dna_mate):
        pass
    #
    def addTelomere(self):
        if self._len_dna > self._max_dna_len:
            self._dna = self._dna[0:self._max_dna_len]
        elif self._len_dna < self._max_dna_len:
            self._dna = np.append( self._dna, np.ones(self._max_dna_len - self._len_dna) * -1 ).astype(int)
        #
    #
#

#
