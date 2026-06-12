#! /usr/bin/python3

##--------------------------------------------------------------------\
#   de_python
#   './de_python/src/differential_evolution.py'
#   Differential Evolution (DE) class. Storn & Price (1997). An
#       evolutionary algorithm modified for the AntennaCAT optimizer set. 
# 
#       Where the basic GA example breeds offspring by crossover + random mutation, 
#       DE builds each trial vector from the scaled DIFFERENCE of randomly
#       chosen population members, so the mutation step size self-adapts
#       to the population's spread in each dimension.
#
#       Follows the AntennaCAT pso_basic / genetic_python template:
#       decoupled step()/call_objective(), one objective evaluation per
#       controller loop pass. Synchronous (classic) DE: all trial
#       vectors for a generation are staged from the current population
#       when the individual counter wraps; each loop pass evaluates one
#       trial; step() consumes the result with greedy parent-vs-trial
#       selection.
#
#       Strategies (STRATEGY in opt_df):
#         1 = DE/rand/1/bin            (classic, most robust)
#         2 = DE/best/1/bin            (greedier, faster on unimodal)
#         3 = DE/current-to-best/1/bin (compromise)
#
#   Author(s): Lauren Linkous, (template: Jonathan Lundquist)
#   Last update: June 11, 2026
##--------------------------------------------------------------------\

import numpy as np
from numpy.random import Generator, MT19937
import sys
np.seterr(all='raise')


class de:
    # arguments should take the form:
    # de([[float, float, ...]], [[float, float, ...]], [[float, ...]], float, int,
    # func, func,
    # dataFrame,
    # class obj,
    # bool, [int, int, ...],
    # int)
    #
    # opt_df contains class-specific tuning parameters
    # POPULATION_SIZE: int. number of vectors NP. typical: 5*n to 10*n,
    #                  minimum 4 (the mutation draws 3 distinct others).
    # MUTATION_F: float. differential weight F in (0, 2]. typical 0.5-0.8
    # CROSS_RATE: float. crossover probability CR in [0, 1]. typical 0.7-0.9
    #                  (name matches genetic_python's CROSS_RATE)
    # STRATEGY: int. 1 = rand/1/bin, 2 = best/1/bin, 3 = current-to-best/1/bin
    # BOUNDARY: int. 1 = random, 2 = reflecting, 3 = absorbing
    #                  (per-component repair, matching the swarm set's options)
    #

    def __init__(self, lbound, ubound, targets, E_TOL, maxit,
                 obj_func, constr_func,
                 opt_df,
                 parent=None,
                 evaluate_threshold=False, obj_threshold=None,
                 decimal_limit=4):

        # Optional parent class func call to write out values that trigger constraint issues
        self.parent = parent

        self.number_decimals = int(decimal_limit)  # limit the number of decimals
                                              # used in cases where real life has limitations on resolution

        # evaluation method for targets
        # True: Evaluate as true targets
        # False: Evaluate as thresholds based on information in obj_threshold
        if evaluate_threshold == False:
            self.evaluate_threshold = False
            self.obj_threshold = None
        else:
            if not(len(obj_threshold) == len(targets)):
                self.debug_message_printout("WARNING: THRESHOLD option selected. +\
                Dimensions for THRESHOLD do not match TARGET array. Defaulting to TARGET search.")
                self.evaluate_threshold = False
                self.obj_threshold = None
            else:
                self.evaluate_threshold = evaluate_threshold  # bool
                self.obj_threshold = np.array(obj_threshold).reshape(-1, 1)  # np.array

        # unpack the opt_df standardized vals
        NO_OF_VECTORS = int(opt_df['POPULATION_SIZE'][0])
        self.mutation_F = float(opt_df['MUTATION_F'][0])
        self.cross_rate = float(opt_df['CROSS_RATE'][0])
        self.strategy = int(opt_df['STRATEGY'][0])
        self.boundary = int(opt_df['BOUNDARY'][0])

        if NO_OF_VECTORS < 4:
            self.debug_message_printout("WARNING: DE requires a population of at \
                least 4 (mutation draws 3 distinct other vectors). Setting to 4.")
            NO_OF_VECTORS = 4

        # optimizer init:
        heightl = np.shape(lbound)[0]
        widthl = np.shape(lbound)[1]
        heightu = np.shape(ubound)[0]
        widthu = np.shape(ubound)[1]

        lbound = np.array(lbound[0], dtype=float)
        ubound = np.array(ubound[0], dtype=float)

        self.rng = Generator(MT19937())

        if ((heightl > 1) and (widthl > 1)) \
           or ((heightu > 1) and (widthu > 1)) \
           or (heightu != heightl) \
           or (widthl != widthu):

            if self.parent == None:
                pass
            else:
                self.parent.debug_message_printout("Error lbound and ubound must be 1xN-dimensional \
                                                        arrays with the same length")

        else:

            self.lbound = lbound
            self.ubound = ubound
            self.n = int(len(lbound))

            '''
            self.M                  : (NP, n) current population (the parents).
            self.F_pop              : (NP,)  scalar fitness per parent (norm of Flist).
            self.F_pop_list         : (NP, out) Flist per parent.
            self.Trials             : (NP, n) staged trial vectors for this generation.
            self.output_size        : An integer value for the output size of obj func.
            self.Gb                 : Global best position.
            self.F_Gb               : Fitness value corresponding to the global best position.
            self.targets            : Target values for the optimization process.
            self.maxit              : Maximum number of OBJECTIVE CALLS.
            self.E_TOL              : Error tolerance.
            self.obj_func           : Objective function to be optimized.
            self.constr_func        : Constraint function.
            self.iter               : Objective function call count.
            self.current_individual : Index of the trial being evaluated this pass.
            self.generation         : Completed generation count.
            self.init_generation    : True while evaluating the initial population.
            self.allow_update       : Flag indicating whether to allow updates.
            self.Flist              : Fitness (distance) of the most recent evaluation.
            self.Fvals              : Raw objective outputs of the most recent evaluation.
            '''

            self.output_size = len(targets)
            self.targets = np.array(targets).reshape(-1, 1)
            self.maxit = maxit
            self.E_TOL = E_TOL
            self.obj_func = obj_func
            self.constr_func = constr_func
            self.iter = 0
            self.allow_update = 0
            self.Flist = []
            self.Fvals = []
            self.number_of_vectors = NO_OF_VECTORS
            self.current_individual = 0
            self.generation = 0
            self.init_generation = True

            # initial population: random feasible points (respawn on
            # constraint violation, like the swarm template)
            variation = ubound - lbound
            self.M = np.zeros((NO_OF_VECTORS, self.n))
            for i in range(NO_OF_VECTORS):
                tries = 0
                while tries < 100:
                    tries = tries + 1
                    x = np.round(self.rng.random(self.n) * variation + lbound,
                                 self.number_decimals)
                    if self.constr_func(x):
                        break
                self.M[i] = x

            self.F_pop = sys.maxsize * np.ones(NO_OF_VECTORS)
            self.F_pop_list = sys.maxsize * np.ones((NO_OF_VECTORS, self.output_size))

            # generation 0 'trials' are the parents themselves: the first
            # NP loop passes evaluate the initial population.
            self.Trials = self.M.copy()

            # global best
            self.Gb = sys.maxsize * np.ones(self.n)
            self.F_Gb = sys.maxsize * np.ones((1, self.output_size))

            self.debug_message_printout("DE successfully initialized")

    def debug_message_printout(self, msg):
        if self.parent == None:
            pass
        else:
            self.parent.debug_message_printout(msg)

    def objective_function_evaluation(self, Fvals, targets):
        # pass in the Fvals & targets so that it's easier to track bugs
        # identical to the pso_basic implementation.
        epsilon = np.finfo(float).eps

        Flist = np.zeros(len(Fvals))

        if self.evaluate_threshold == True:  # THRESHOLD
            ctr = 0
            for i in targets:
                o_thres = int(self.obj_threshold[ctr].item())
                t = targets[ctr].item()
                fv = Fvals[ctr].item()

                if o_thres == 0:  # TARGET. default
                    Flist[ctr] = abs(t - fv)
                elif o_thres == 1:  # LESS THAN OR EQUAL
                    if fv <= t:
                        Flist[ctr] = epsilon
                    else:
                        Flist[ctr] = abs(t - fv)
                elif o_thres == 2:  # GREATER THAN OR EQUAL
                    if fv >= t:
                        Flist[ctr] = epsilon
                    else:
                        Flist[ctr] = abs(t - fv)
                else:
                    self.debug_message_printout("ERROR: unrecognized threshold value. Evaluating as TARGET")
                    Flist[ctr] = abs(t - fv)
                ctr = ctr + 1
        else:  # TARGET as default
            Flist = abs(targets - Fvals)

        return Flist

    def call_objective(self, allow_update):
        # evaluates the objective function at the current TRIAL vector AND
        # computes the target/threshold fitness. After this returns,
        # get_latest_eval() and converged() reflect the evaluation that
        # just happened, BEFORE the next step() consumes it (selection)
        # and advances.
        newFVals, noError = self.obj_func(self.Trials[self.current_individual],
                                          self.output_size)
        if noError == True:
            self.Fvals = np.array(newFVals).reshape(-1, 1)
            if allow_update:
                # EVALUATE OBJECTIVE FUNCTION - TARGET OR THRESHOLD
                self.Flist = self.objective_function_evaluation(self.Fvals, self.targets)
                self.iter = self.iter + 1
                self.allow_update = 1
            else:
                self.allow_update = 0
        return noError  # return is for error reporting purposes only

    def step(self, suppress_output):
        if not suppress_output:
            msg = "\n-----------------------------\n" + \
                "STEP #" + str(self.iter) + "\n" + \
                "-----------------------------\n" + \
                "Generation:\n" + \
                str(self.generation) + "\n" + \
                "Current Individual:\n" + \
                str(self.current_individual) + "\n" + \
                "Current Trial Vector:\n" + \
                str(self.Trials[self.current_individual]) + "\n" + \
                "Parent Fitness (norm):\n" + \
                str(self.F_pop[self.current_individual]) + "\n" + \
                "-----------------------------"
            self.debug_message_printout(msg)

        if self.allow_update:
            # 1. CONSUME the previously evaluated trial (fresh data from
            #    the last call_objective): greedy selection. the trial
            #    replaces its parent if it is at least as good. <= (not <)
            #    is the standard DE choice - it lets the population drift
            #    across plateaus.
            i = self.current_individual
            trial_norm = np.linalg.norm(self.Flist)
            if trial_norm <= self.F_pop[i]:
                self.M[i] = self.Trials[i]
                self.F_pop[i] = trial_norm
                self.F_pop_list[i] = np.squeeze(np.asarray(self.Flist))
                if trial_norm < np.linalg.norm(self.F_Gb):
                    self.F_Gb = np.array([np.atleast_1d(np.squeeze(np.asarray(self.Flist)))])
                    self.Gb = 1 * self.Trials[i]

            # 2. ADVANCE to the next individual. when the counter wraps,
            #    the generation is complete: stage a full set of new trial
            #    vectors from the (just-updated) population.
            self.current_individual = self.current_individual + 1
            if self.current_individual == self.number_of_vectors:
                self.current_individual = 0
                self.generation = self.generation + 1
                self.init_generation = False
                self.stage_trials()

            if self.complete() and not suppress_output:
                msg = "\nOPTIMIZATION COMPLETE:\nPoints: \n" + str(self.Gb) + "\n" + \
                    "Iterations: \n" + str(self.iter) + "\n" + \
                    "Generations: \n" + str(self.generation) + "\n" + \
                    "Flist: \n" + str(self.F_Gb) + "\n" + \
                    "Norm Flist: \n" + str(np.linalg.norm(self.F_Gb)) + "\n"
                self.debug_message_printout(msg)

    def stage_trials(self):
        # build one trial vector per parent via differential mutation +
        # binomial crossover. SYNCHRONOUS DE: every mutant is built from
        # the population as it stands at the start of the generation.
        best = int(np.argmin(self.F_pop))
        for i in range(self.number_of_vectors):
            tries = 0
            while tries < 100:
                tries = tries + 1
                trial = self.make_trial(i, best)
                if self.constr_func(trial):
                    break
            else:
                # constraint never satisfied: random feasible respawn,
                # like the swarm template
                variation = self.ubound - self.lbound
                while True:
                    trial = np.round(self.rng.random(self.n) * variation + self.lbound,
                                     self.number_decimals)
                    if self.constr_func(trial):
                        break
            self.Trials[i] = trial

    def make_trial(self, i, best):
        # choose 3 distinct population members, all different from i
        choices = [k for k in range(self.number_of_vectors) if k != i]
        r1, r2, r3 = self.rng.choice(choices, size=3, replace=False)

        # differential mutation
        if self.strategy == 2:    # DE/best/1
            mutant = self.M[best] + self.mutation_F * (self.M[r1] - self.M[r2])
        elif self.strategy == 3:  # DE/current-to-best/1
            mutant = self.M[i] \
                + self.mutation_F * (self.M[best] - self.M[i]) \
                + self.mutation_F * (self.M[r1] - self.M[r2])
        else:                     # DE/rand/1 (default)
            mutant = self.M[r1] + self.mutation_F * (self.M[r2] - self.M[r3])

        # binomial crossover. jrand guarantees the trial inherits at
        # least one mutant component (otherwise trial == parent).
        trial = self.M[i].copy()
        jrand = int(self.rng.integers(0, self.n))
        for j in range(self.n):
            if (self.rng.random() < self.cross_rate) or (j == jrand):
                trial[j] = mutant[j]

        # per-component bounds repair, matching the swarm set's options
        for j in range(self.n):
            if (trial[j] < self.lbound[j]) or (trial[j] > self.ubound[j]):
                if self.boundary == 2:    # reflecting
                    if trial[j] < self.lbound[j]:
                        trial[j] = self.lbound[j] + (self.lbound[j] - trial[j])
                    else:
                        trial[j] = self.ubound[j] - (trial[j] - self.ubound[j])
                    # reflection can overshoot on large violations
                    trial[j] = min(max(trial[j], self.lbound[j]), self.ubound[j])
                elif self.boundary == 3:  # absorbing (clamp)
                    trial[j] = min(max(trial[j], self.lbound[j]), self.ubound[j])
                else:                     # random (default)
                    trial[j] = self.lbound[j] + self.rng.random() \
                        * (self.ubound[j] - self.lbound[j])

        return np.round(trial, self.number_decimals)

    # funcs from other optimizers in the AntennaCAT set for stop conditions

    def get_latest_eval(self):
        # L2 norm of the fitness of the MOST RECENTLY evaluated trial
        # (pending, not yet consumed by step()'s selection).
        if self.allow_update and np.shape(self.Flist)[0] > 0:
            return np.linalg.norm(self.Flist)
        return None

    def converged(self):
        convergence = np.linalg.norm(self.F_Gb) < self.E_TOL
        return convergence

    def maxed(self):
        max_iter = self.iter >= self.maxit
        return max_iter

    def complete(self):
        done = self.converged() or self.maxed()
        return done

    def get_convergence_data(self):
        best_eval = np.linalg.norm(self.F_Gb)
        return self.iter, best_eval

    def get_optimized_soln(self):
        return np.vstack(self.Gb)

    def get_optimized_outs(self):
        return np.vstack(np.atleast_1d(np.squeeze(self.F_Gb)))

    # for plotting
    def get_search_locations(self):
        return self.M

    def get_fitness_values(self):
        return self.F_pop_list

    def absolute_mean_deviation_of_population(self):
        # population spread metric, analogous to the PSO version. useful
        # as a stagnation diagnostic: DE's step sizes scale with this.
        abs_mean_dev = np.mean(np.abs(self.M - np.mean(self.M, axis=0)), axis=0)
        return np.linalg.norm(abs_mean_dev)

    def export_de(self):
        de_export = {
            'evaluate_threshold': [self.evaluate_threshold],
            'obj_threshold': [self.obj_threshold],
            'targets': [self.targets],
            'lbound': [self.lbound],
            'ubound': [self.ubound],
            'output_size': [self.output_size],
            'maxit': [self.maxit],
            'E_TOL': [self.E_TOL],
            'mutation_F': [self.mutation_F],
            'cross_rate': [self.cross_rate],
            'strategy': [self.strategy],
            'boundary': [self.boundary],
            'number_of_vectors': [self.number_of_vectors],
            'iter': [self.iter],
            'generation': [self.generation],
            'init_generation': [self.init_generation],
            'current_individual': [self.current_individual],
            'allow_update': [self.allow_update],
            'M': [self.M],
            'F_pop': [self.F_pop],
            'F_pop_list': [self.F_pop_list],
            'Trials': [self.Trials],
            'Gb': [self.Gb],
            'F_Gb': [self.F_Gb],
            'Flist': [self.Flist],
            'Fvals': [self.Fvals]}
        return de_export

    def import_de(self, de_export, obj_func):
        self.evaluate_threshold = de_export['evaluate_threshold'][0]
        self.obj_threshold = de_export['obj_threshold'][0]
        self.targets = de_export['targets'][0]
        self.lbound = de_export['lbound'][0]
        self.ubound = de_export['ubound'][0]
        self.output_size = de_export['output_size'][0]
        self.maxit = de_export['maxit'][0]
        self.E_TOL = de_export['E_TOL'][0]
        self.mutation_F = de_export['mutation_F'][0]
        self.cross_rate = de_export['cross_rate'][0]
        self.strategy = de_export['strategy'][0]
        self.boundary = de_export['boundary'][0]
        self.number_of_vectors = de_export['number_of_vectors'][0]
        self.iter = de_export['iter'][0]
        self.generation = de_export['generation'][0]
        self.init_generation = de_export['init_generation'][0]
        self.current_individual = de_export['current_individual'][0]
        self.allow_update = de_export['allow_update'][0]
        self.M = de_export['M'][0]
        self.F_pop = de_export['F_pop'][0]
        self.F_pop_list = de_export['F_pop_list'][0]
        self.Trials = de_export['Trials'][0]
        self.Gb = de_export['Gb'][0]
        self.F_Gb = de_export['F_Gb'][0]
        self.Flist = de_export['Flist'][0]
        self.Fvals = de_export['Fvals'][0]
        self.n = int(len(self.lbound))
        self.obj_func = obj_func
