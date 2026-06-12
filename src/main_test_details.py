#! /usr/bin/python3

##--------------------------------------------------------------------\
#   de_python
#   './de_python/src/main_test_details.py'
#   Test function/example for using the DE optimizer. Matches the
#       format of the other optimizers in the AntennaCAT suite.
#
#   Author(s): Lauren Linkous, Jonathan Lundquist
#   Last update: March 12, 2025
##--------------------------------------------------------------------\


import sys
import time
import pandas as pd
import numpy as np

try:  # for outside func calls, program calls
    sys.path.insert(0, './de_python/src/')
    from differential_evolution import de
except:  # for local, unit testing
    from differential_evolution import de



# OBJECTIVE FUNCTION SELECTION
#import one_dim_x_test.configs_F as func_configs     # single objective, 1D input
#import himmelblau.configs_F as func_configs         # single objective, 2D input
import lundquist_3_var.configs_F as func_configs     # multi objective function



class TestDetails():
    def __init__(self):
        # Constant variables
        E_TOL = 10 ** -4      # Convergence Error Tolerance
        MAXIT = 5000          # Maximum allowed objective function calls

        # Objective function dependent variables
        LB = func_configs.LB              # Lower boundaries, [[-5, -5]]
        UB = func_configs.UB              # Upper boundaries, [[5, 5]]
        TARGETS = func_configs.TARGETS    # Target values for output

        # threshold is same dims as TARGETS
        # 0 = use target value as actual target. value should EQUAL target
        # 1 = use as threshold. value should be LESS THAN OR EQUAL to target
        # 2 = use as threshold. value should be GREATER THAN OR EQUAL to target
        # DEFAULT THRESHOLD
        THRESHOLD = np.zeros_like(TARGETS)
        evaluate_threshold = False

        # Objective function dependent variables
        func_F = func_configs.OBJECTIVE_FUNC   # objective function
        constr_F = func_configs.CONSTR_FUNC    # constraint function

        # optimizer specific vars
        POPULATION_SIZE = 20  # number of vectors NP. typical 5*n to 10*n, min 4
        MUTATION_F = 0.7      # differential weight F in (0, 2]
        CROSS_RATE = 0.8      # crossover probability CR in [0, 1]
        STRATEGY = 1          # 1 = rand/1/bin, 2 = best/1/bin, 3 = current-to-best/1/bin
        BOUNDARY = 1          # 1 = random, 2 = reflecting, 3 = absorbing

        # optimizer setting values
        parent = None
        self.suppress_output = True
        self.allow_update = True

        self.best_eval = 1

        # instantiation of DE optimizer
        opt_params = {'POPULATION_SIZE': [POPULATION_SIZE],
                    'MUTATION_F': [MUTATION_F],
                    'CROSS_RATE': [CROSS_RATE],
                    'STRATEGY': [STRATEGY],
                    'BOUNDARY': [BOUNDARY]}
        opt_df = pd.DataFrame(opt_params)
        self.myOptimizer = de(LB, UB, TARGETS, E_TOL, MAXIT,
                func_F, constr_F,
                opt_df,
                parent=parent,
                evaluate_threshold=evaluate_threshold,
                obj_threshold=THRESHOLD)


    def debug_message_printout(self, txt):
        if txt is None:
            return
        # sets the string as it gets it
        curTime = time.strftime("%H:%M:%S", time.localtime())
        msg = "[" + str(curTime) +"] " + str(txt)
        print(msg)


    def run(self):

        # instantiation of particle swarm optimizer 
        while not self.myOptimizer.complete():

            # step through optimizer processing
            self.myOptimizer.step(self.suppress_output)

            # call the objective function, control 
            # when it is allowed to update and return 
            # control to optimizer
            self.myOptimizer.call_objective(self.allow_update)
            iter, eval = self.myOptimizer.get_convergence_data()
            if (eval < self.best_eval) and (eval != 0):
                self.best_eval = eval
            if self.suppress_output:
                if iter%100 ==0: #print out every 100th iteration update
                    print("Iteration")
                    print(iter)
                    print("Best Eval")
                    print(self.best_eval)

        print("Optimized Solution")
        print(self.myOptimizer.get_optimized_soln())
        print("Optimized Outputs")
        print(self.myOptimizer.get_optimized_outs())



if __name__ == "__main__":
    pso = TestDetails()
    pso.run()
