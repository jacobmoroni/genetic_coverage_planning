from pathmaker import PathMaker
from IPython.core.debugger import set_trace
from importlib import reload

import numpy as np
import copy
from tqdm import tqdm

import gori_tools as got
import pathmaker
reload(got)
reload(pathmaker)


class GeneticalGorithm():
    def __init__(self, mappy, pather, gen_params):
        # setup params
        self._G_sz = gen_params['gen_size']
        self._starting_path_len = gen_params['starting_path_len']
        self._num_agents = gen_params['num_agents']
        self._gamma = gen_params['gamma']
        self._cov_constr_0 = -gen_params['coverage_constr_0']
        self._cov_constr_f = -gen_params['coverage_constr_f']
        self._cov_aging = gen_params['coverage_aging']
        org_params = gen_params['org_params']

        # initialize generation to 0
        self._gen_num = 0

        # list for holding all chromosomes in parent generation
        self._gen_parent = []
        self._gen_parent_fit = []

        # create list of first generation
        self.firstGeneration(mappy, pather, org_params)

    def firstGeneration(self, mappy, pather, org_params):
        # create random set of chromosomes
        start_idx = org_params['start_idx']
        # pool this for loop with multiprocess
        adeveing = tqdm(total=self._G_sz,
                        desc="Adeves (aka. Adam and Eve'ing)")
        while len(self._gen_parent) < self._G_sz:
            rand_paths = []
            for agent in range(self._num_agents):
                rand_path = pather.makeMeAPath(self._starting_path_len,
                                               start_idx[agent])
                rand_paths.append(rand_path)
            cand_org = Organism(rand_paths, mappy, pather, org_params)
            if cand_org._obj_val[0] <= self._cov_constr_0:
                self._gen_parent.append(cand_org)
                adeveing.update(1)
        self._gen_parent_fit = self.rackNstack(self._gen_parent)

    def runEvolution(self, num_generations):
        # list for holding all chromosomes in children generation
        pareto_1_gen = []
        for _ in tqdm(range(num_generations), desc="Evolving"):
            gen_child = []
            # grab a set of parents that are good to reproduce
            strong_parents = got.lowVarSample(self._gen_parent,
                                              self._gen_parent_fit,
                                              self._gamma)
            for mommy, daddy in zip(strong_parents[:self._G_sz//2],
                                    strong_parents[self._G_sz//2:]):
                # do crossover with all the worthy parents
                family_kids = mommy.crossover(daddy)
                gen_child += family_kids

            # only the strong will survive this arena
            # it's dog eat dog, parent eat child. Animals.
            self.arena(self._gen_parent + gen_child)
            self._gen_num += 1

            pareto_1_gen += [got.getObjValsList(self)]
        return pareto_1_gen

    def arena(self, gladiators):
        # rack and stack
        gladiators_fit = self.rackNstack(gladiators)
        gladiators, gladiators_fit = got.sortBy(gladiators, gladiators_fit)
        # kill the unfit ones
        self._gen_parent = gladiators[:self._G_sz]
        self._gen_parent_fit = gladiators_fit[:self._G_sz]

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
        alpha = min(1, (self._gen_num/self._cov_aging))
        coverage_constr = (1-alpha)*self._cov_constr_0 + \
            alpha*self._cov_constr_f
        for ii, glad in enumerate(gladiators):
            if glad._obj_val[0] > coverage_constr:
                gladiators[ii]._obj_val_sc[0] = glad._obj_val[0]
                gladiators[ii]._obj_val_sc[1] = glad._obj_val[1] + 1000.0
            else:
                gladiators[ii]._obj_val_sc[0] = glad._obj_val[0]
                gladiators[ii]._obj_val_sc[1] = glad._obj_val[1]

        for ii, gen_fit_1 in enumerate(gladiators):
            comp_min = []
            for jj, gen_fit_2 in enumerate(gladiators):
                if ii == jj:
                    continue
                else:
                    comp_min.append(np.min([gen_fit_1._obj_val_sc[0] -
                                            gen_fit_2._obj_val_sc[0],
                                            gen_fit_1._obj_val_sc[1] -
                                            gen_fit_2._obj_val_sc[1]]))

            comp_max.append(np.max(comp_min))

        fitnesses = comp_max
        return fitnesses


class Organism():
    def __init__(self, dna_list, mappy, pather, org_params):
        self._max_dna_len = org_params['max_dna_len']
        self._min_dna_len = org_params['min_dna_len']

        self._xover_probability = org_params['crossover_prob']
        self._time_thresh = org_params['crossover_time_thresh']
        self._mutate_probability = org_params['mutation_prob']
        self._muterpolate_probability = org_params['muterpolate_prob']
        self._num_muterpolations = org_params['num_muterpolations']
        self._srch_dist = org_params['muterpolation_srch_dist']
        self._P_muterpolate_each = org_params['muterpolation_sub_prob']

        self._min_solo_lcs = org_params['min_solo_lcs']
        self._min_comb_lcs = org_params['min_comb_lcs']
        self._ft_scale = org_params['flight_time_scale']

        self._org_params = org_params
        self._pather = pather
        self._mappy = mappy
        self._scale = self._mappy._scale
        self._narrowest_hall = self._mappy._hall_width
        self._safety_buffer = self._mappy._safety_buffer

        self._dna_list = dna_list
        self._num_agents = len(self._dna_list)
        self._len_dna = (np.ones(self._num_agents)).astype(int)
        for agent in range(self._num_agents):
            self._len_dna[agent] = len(self._dna_list[agent])

        self.addTelomere()

        # apply mutation
        do_it = np.random.rand()
        if do_it < self._mutate_probability:
            self.mutation()

        do_it = np.random.rand()
        if do_it < self._muterpolate_probability:
            self.muterpolate()

        #prune u-turns
        self.pruneUTurns()

        # compute values of both objectives
        self._obj_val = self.calcObj()
        self._obj_val_sc = [None, None]
        for agent in range(self._num_agents):
            self._len_dna[agent] = len(self._dna_list[agent])
        self.addTelomere()

    def calcObj(self):
        coverage, travel_dist = self._mappy.getCoverage(self._dna)
        lc_mat = self._mappy.getLoopClosures(self._dna)
        # total number of solo_loop closures
        solo_lcs = np.diagonal(lc_mat)
        combo_lc_mat = lc_mat-np.diag(solo_lcs)
        # total number of combined loop closures
        combo_lcs = np.sum(combo_lc_mat)

        # computing number of connected components using adjacency matrix,
        # degree matrix, and grapha laplacian
        adjacency = ((combo_lc_mat + combo_lc_mat.T) >= 1).astype(int)
        degree = np.diag(np.sum(adjacency, axis=0))
        graph_laplacian = degree - adjacency
        eig, _ = np.linalg.eig(graph_laplacian)
        num_con = self._num_agents - np.sum(eig > 1e-6)

        travel_dist = travel_dist * self._ft_scale
        # apply contstraints by zeroing coverage
        if ((solo_lcs < self._min_solo_lcs).any()
                or combo_lcs < self._min_comb_lcs or num_con) > 1:
            coverage = 0
        return [coverage, travel_dist]

    def padMinDNA(self, dna):
        dna = dna[dna != -1]
        if len(dna) < self._min_dna_len:
            len_tail = self._min_dna_len - len(dna)
            dna_tail = self._pather.makeMeAPath(len_tail, dna[-1])
            dna = np.append(dna, dna_tail[1:])
        return dna

    def crossover(self, mate):
        uniform_num = np.random.rand(self._num_agents)
        xover_pts = self.matchWaypt(mate._dna, mate._len_dna)
        new_dna1 = []
        new_dna2 = []
        for agent in range(self._num_agents):
            if (uniform_num[agent] < self._xover_probability
                    and len(xover_pts[agent]) > 0):

                xover_idx = np.arange(len(xover_pts[agent]))
                single_idx = np.random.choice(xover_idx)

                single_pt = xover_pts[agent][single_idx]

                dna1 = self._dna[agent][0:single_pt[0]]
                dna2 = mate._dna[agent][0:single_pt[1]]

                dna1 = np.append(dna1, mate._dna[agent][single_pt[1]:])
                dna2 = np.append(dna2, self._dna[agent][single_pt[0]:])
                dna1 = self.padMinDNA(dna1)
                dna2 = self.padMinDNA(dna2)

            else:
                dna1 = self._dna[agent]
                dna2 = mate._dna[agent]
            dna1 = dna1[dna1 != -1]
            dna2 = dna2[dna2 != -1]
            new_dna1.append(dna1)
            new_dna2.append(dna2)
        lil_timmy = Organism(new_dna1,
                             self._mappy,
                             self._pather,
                             self._org_params)
        lil_susy = Organism(new_dna2,
                            self._mappy,
                            self._pather,
                            self._org_params)

        return [lil_timmy, lil_susy]

    def mutation(self):
        for agent in range(self._num_agents):
            idx = np.random.randint(1, self._len_dna[agent]-1)
            len_tail = self._len_dna[agent] - idx
            try:
                if (self._len_dna[agent]+1) < self._max_dna_len:
                    len_tail += np.random.randint(self._max_dna_len -
                                                  self._len_dna[agent])
            except:
                set_trace()
            dna_tail = self._pather.makeMeAPath(len_tail,
                                                self._dna[agent, idx],
                                                self._dna[agent, 0:idx+1])
            dna_head = self._dna_list[agent][0:idx]

            self._dna_list[agent] = np.append(dna_head, dna_tail)
            self._len_dna[agent] = len(self._dna_list[agent])
        self.addTelomere()

    def muterpolate(self):
        for agent in range(self._num_agents):
            try:
                mut_points = np.random.choice(np.arange(0,
                                                        self._len_dna[agent]-(self._srch_dist+3), 1),
                                              self._num_muterpolations)
                # reverse sort points to minimize chance of trying to access
                # out of bounds after shortenning path
                mut_points = np.sort(mut_points)[::-1]
            except:
                print("DNA is getting too short")
                mut_points = []
            for idx1 in mut_points:
                for idx2 in range(idx1+self._srch_dist+1, idx1+1, -1):
                    do_it = np.random.rand()
                    if do_it < self._P_muterpolate_each:
                        try:
                            pt1 = self._dna_list[agent][idx1]
                            pt2 = self._dna_list[agent][idx2]
                        except:
                            break
                        if self._pather._graph[pt1, pt2]:
                            # check to see if we can get there directly
                            np.delete(self._dna_list[agent],
                                      np.arange(idx1+1, idx2-1, 1))
                            break
                        else:
                            # check the mutual traversible space
                            options = np.where(
                                self._pather._graph[pt1]*self._pather._graph[pt2])[0]
                            if len(options) > 0:
                                step = np.random.choice(options)
                                self._dna_list[agent][idx1+1] = step
                                self._dna_list[agent] = np.delete(
                                    self._dna_list[agent],
                                    np.arange(idx1+2, idx2-1, 1))
                                break
            self._dna_list[agent] = self._dna_list[agent][self._dna_list[agent] != -1]
            self._len_dna[agent] = len(self._dna_list[agent])
        self.addTelomere()

    def matchWaypt(self, dna_mate, len_dna_mate):
        keepout_idx = 1
        xover_pts = []
        for agent in range(self._dna.shape[0]):
            dna_poss = self._dna[agent][self._dna[agent] != -1]
            mate_poss = dna_mate[agent][dna_mate[agent] != -1]
            # broadcast is brd
            brd_diff_mat = np.abs(dna_poss[keepout_idx:self._len_dna[agent]+1, None] -
                                  mate_poss[None, keepout_idx:len_dna_mate[agent]+1])
            un_pruned_pts = np.array(
                np.where(brd_diff_mat == 0)).T + keepout_idx
            xover_tf = np.abs(
                un_pruned_pts[:, 0] - un_pruned_pts[:, 1]) < self._time_thresh
            xover_pts.append(un_pruned_pts[xover_tf])

        return xover_pts

    def addTelomere(self):
        for agent in range(self._num_agents):
            if self._len_dna[agent] > self._max_dna_len:
                self._dna_list[agent] = self._dna_list[agent][0:self._max_dna_len]
                self._len_dna[agent] = self._max_dna_len
            elif self._len_dna[agent] < self._max_dna_len:
                self._dna_list[agent] = np.append(self._dna_list[agent],
                                                  np.ones(self._max_dna_len - \
                                                  self._len_dna[agent]) * -1).astype(int)
        self._dna = np.vstack(self._dna_list)

    def pruneUTurns(self):
        pruned = False
        for agent in range(self._num_agents):
            _, dups = self._mappy.getDuplicateWPs(self._dna[agent][self._dna[agent]!=-1])
            for dup in dups:
                dup = np.sort(dup)
                diffs = np.diff(dup)
                for ii,diff in enumerate(diffs):
                    if diff <= 2:
                        self._dna[agent,dup[ii]:dup[ii+1]] = -1
                        set_trace()
                        pruned = True
            if pruned == True:
                set_trace()
            self._dna_list[agent] = self._dna[agent][self._dna[agent] != -1]
            self._len_dna[agent] = len(self._dna_list[agent])
        self.addTelomere()
        if pruned == True:
            set_trace()
            # set_trace()
            #TODO: Work on this to prune away waypoints that are 1 or 2 away from eachother. 
            # try to do it vectorized
            # for dup in dups:
                
