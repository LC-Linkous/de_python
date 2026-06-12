# de_python

Simple Differential Evolution (DE) optimizer written in Python. Built on the [pso_basic](https://github.com/jonathan46000/pso_python) / genetic_python template for data collection baseline. This implementation uses the decoupled `step()` / `call_objective()` structure so that exactly one objective evaluation happens per controller loop pass, and termination conditions can be checked against fresh data.

de_python has been updated to increase modularity with the optimizer suite collection.

## Table of Contents
* [Differential Evolution](#differential-evolution)
    * [Mutation Strategies](#mutation-strategies)
* [Requirements](#requirements)
* [Implementation](#implementation)
    * [Initialization](#initialization) 
    * [State Machine-based Structure](#state-machine-based-structure)
    * [Importing and Exporting Optimizer State](#importing-and-exporting-optimizer-state)
    * [Constraint Handling](#constraint-handling)
    * [Boundary Types](#boundary-types)
    * [Multi-Objective Optimization](#multi-objective-optimization)
    * [Objective Function Handling](#objective-function-handling)
      * [Creating a Custom Objective Function](#creating-a-custom-objective-function)
      * [Internal Objective Function Example](#internal-objective-function-example)
    * [Target vs. Threshold Configuration](#target-vs-threshold-configuration)
* [Example Implementations](#example-implementations)
    * [Basic DE Example](#basic-de-example)
    * [Detailed Messages](#detailed-messages)
    * [Realtime Graph](#realtime-graph)
* [References](#references)
* [Related Publications and Repositories](#related-publications-and-repositories)
* [Licensing](#licensing)  

## Differential Evolution

Differential Evolution (DE) is a popular evolutionary optimization algorithm introduced in "Differential Evolution – A Simple and Efficient Heuristic for Global Optimization over Continuous Spaces" [1] (R. Storn & K. Price, 1997). It is an evolutionary algorithm with a population, mutation, crossover, and selection — but where a genetic algorithm mutates with random perturbations of fixed character, DE builds each mutant from the scaled difference of randomly chosen population members.

DE maintains a population of `NP` parent vectors. Each generation, one **trial vector** is built per parent (mutation + binomial crossover) and evaluated, then greedy one-to-one selection decides which survives. Because mutation steps are built from population differences, the step size is self-adapting: when the population is spread out, steps are large (exploration). As it contracts into a basin, steps shrink with it (refinement) based per dimension, with no schedule, no temperature, and no step-size parameter to tune down. 

Selection is `positional`. A trial competes only with its own parent (not the rest of the population) which preserves diversity. The trial replaces its parent if it is at least as good (`<=`, the standard DE choice — it lets the population drift across plateaus).

### Mutation Strategies

A mutant is assembled from population members (r1, r2, r3 distinct from each other and from the parent i), scaled by the differential weight `F`. Three strategies are included, selected with the `STRATEGY` parameter:

| `STRATEGY` | Name | Mutant | Note |
|---|---|---| --- |
| 1 (default) | DE/rand/1 | `x_r1 + F·(x_r2 − x_r3)`| most robust, most explorative |
| 2 | DE/best/1 | `x_best + F·(x_r1 − x_r2)`| greedier, faster on unimodal problems |
| 3 | DE/current-to-best/1 | `x_i + F·(x_best − x_i) + F·(x_r1 − x_r2)`| compromise |

In binomial crossover, each trial component comes from the mutant with probability `CR`, otherwise from the parent. One randomly chosen index (`jrand`) is always taken from the mutant, guaranteeing the trial differs from its parent.

Tuning rules of thumb: `NP` of 5n–10n (minimum 4 — the mutation draws 3 distinct others), `F` 0.5–0.8, `CR` 0.7–0.9. High `CR` suits non-separable problems (such as coupled antenna parameters); low `CR` suits separable ones. If progress stalls, check `absolute_mean_deviation_of_population()`. 

DE's effective step size scales with population spread, so a collapsed population means restart or raise `F`.

## Requirements

This project requires numpy, pandas, and matplotlib for plotting live optimization. 

To run the optimizer without visualization, only numpy and pandas are requirements.

Use 'pip install -r requirements.txt' to install the following dependencies:

```python
contourpy==1.3.3
cycler==0.12.1
fonttools==4.63.0
kiwisolver==1.5.0
matplotlib==3.10.9
numpy==2.4.6
packaging==26.2
pandas==3.0.3
pillow==12.2.0
pyparsing==3.3.2
python-dateutil==2.9.0.post0
six==1.17.0
tzdata==2026.2

```

Optionally, requirements can be installed manually with:

```python
pip install  matplotlib, numpy, pandas

```


## Implementation

### Initialization 

```python
    # Constant variables
    POPULATION_SIZE = 20         # Number of vectors in the population (NP)
    E_TOL = 10 ** -4             # Convergence Tolerance
    MAXIT = 10000                # Maximum allowed iterations
    BOUNDARY = 1                 # int boundary 1 = random,      2 = reflecting
                                 #              3 = absorbing
                                 # (no invisible option - a trial must be
                                 # evaluable, since selection needs its fitness)

    # Objective function dependent variables
    func_F = func_configs.OBJECTIVE_FUNC  # objective function
    constr_F = func_configs.CONSTR_FUNC   # constraint function

    LB = func_configs.LB              # Lower boundaries, [[0.21, 0, 0.1]]
    UB = func_configs.UB              # Upper boundaries, [[1, 1, 0.5]]   
    OUT_VARS = func_configs.OUT_VARS  # Number of output variables (y-values)
    TARGETS = func_configs.TARGETS    # Target values for output

    # threshold is same dims as TARGETS
    # 0 = use target value as actual target. value should EQUAL target
    # 1 = use as threshold. value should be LESS THAN OR EQUAL to target
    # 2 = use as threshold. value should be GREATER THAN OR EQUAL to target
    #DEFAULT THRESHOLD
    THRESHOLD = np.zeros_like(TARGETS) 

    # optimizer constants
    MUTATION_F = 0.7         # Differential weight F. Typical range 0.5 - 0.8
    CROSS_RATE = 0.8         # Crossover rate CR. Typical range 0.7 - 0.9
    STRATEGY = 1             # int strategy 1 = DE/rand/1
                             #              2 = DE/best/1
                             #              3 = DE/current-to-best/1

    best_eval = 1
    parent = None            # for the optimizer test ONLY
    evaluate_threshold = True # use target or threshold. True = THRESHOLD, False = EXACT TARGET
    suppress_output = True   # Suppress the console output of the optimizer
    allow_update = True      # Allow objective call to update state 

    # Constant variables
    opt_params = {'POPULATION_SIZE': [POPULATION_SIZE], # Number of vectors in population (NP)
                'MUTATION_F': [MUTATION_F],             # Differential weight F
                'CROSS_RATE': [CROSS_RATE],             # Crossover rate CR
                'STRATEGY': [STRATEGY],                 # int strategy 1 = rand/1, 2 = best/1,
                                                        #   3 = current-to-best/1
                'BOUNDARY': [BOUNDARY] }                # int boundary 1 = random, 2 = reflecting
                                                        #   3 = absorbing

    opt_df = pd.DataFrame(opt_params)
    myOptimizer = de(LB, UB, TARGETS, E_TOL, MAXIT,
                            func_F, constr_F,
                            opt_df,
                            parent=parent, 
                            evaluate_threshold=evaluate_threshold, obj_threshold=THRESHOLD,
                            decimal_limit = 4)  

    # arguments should take the form: 
    # de([[float, float, ...]], [[float, float, ...]], [[float, ...]], float, int,
    # func, func,
    # dataFrame,
    # class obj, 
    # bool, [int, int, ...], 
    # int) 
    # 
    # opt_df contains class-specific tuning parameters
    # POPULATION_SIZE: int
    # MUTATION_F: float
    # CROSS_RATE: float
    # STRATEGY: int. 1 = DE/rand/1, 2 = DE/best/1, 3 = DE/current-to-best/1
    # boundary: int. 1 = random, 2 = reflecting, 3 = absorbing

```

**NOTE:** `decimal_limit` quantizes the search space. 

Trials are rounded to `decimal_limit` places, so achievable fitness is floor-limited by the objective's slope at that resolution (e.g., the 1D test function plateaus near 1.5e-5 with the default 4 decimals). 

This matches the suite-wide behavior and is intentional. Raise `decimal_limit` for finer convergence. The optimizer is deterministic given a seeded RNG (`self.rng`); population initialization and all donor/crossover draws come from it.

### State Machine-based Structure

This optimizer uses a state machine structure to control the staging of trial vectors, the call to the objective function, and the greedy selection of survivors. The state machine implementation preserves the initial algorithm while making it possible to integrate other programs, classes, or functions as the objective function.


This optimizer uses a state machine structure to control the movement of the particles, call to the objective function, and the evaluation of current positions. The state machine implementation preserves the initial algorithm while making it possible to integrate other programs, classes, or functions as the objective function.

A controller with a `while loop` to check the completion status of the optimizer drives the process. Completion status is determined by at least 1) a set MAX number of iterations, and 2) the convergence to a given target using the L2 norm.  Iterations are counted by calls to the objective function. 

Within this `while loop` are three function calls to control the optimizer class:
* **complete**: the `complete function` checks the status of the optimizer and if it has met the convergence or stop conditions.
* **step**: the `step function` takes a boolean variable (suppress_output) as an input to control detailed printout on current particle (or agent) status. This function moves the optimizer one step forward.  
* **call_objective**: the `call_objective function` takes a boolean variable (allow_update) to control if the objective function is able to be called. In most implementations, this value will always be true. However, there may be cases where the controller or a program running the state machine needs to assert control over this function without stopping the loop.

Additionally, **get_convergence_data** can be used to preview the current status of the optimizer, including the current best evaluation and the iterations.


This implementation follows the staged-offspring pattern: all trial vectors for a generation are staged when the individual counter wraps (synchronous, classic DE — every mutant is built from the population as it stood at the start of the generation), and the controller loop evaluates one trial per pass. Generation 0 evaluates the parents: the initial trials *are* the initial population, and since parent fitness starts at `sys.maxsize`, the standard selection rule stores them without a special code path.

The code below is an example of this process:

```python
    while not myOptimizer.complete():
        # step through optimizer processing
        # this performs greedy selection on the previous trial
        # and stages the next generation when the counter wraps
        myOptimizer.step(suppress_output)
        # call the objective function, control 
        # when it is allowed to update and return 
        # control to optimizer
        myOptimizer.call_objective(allow_update)
        # check the current progress of the optimizer
        # iter: the number of objective function calls
        # eval: current 'best' evaluation of the optimizer
        iter, eval = myOptimizer.get_convergence_data()
        if (eval < best_eval) and (eval != 0):
            best_eval = eval
        
        # optional. if the optimizer is not printing out detailed 
        # reports, preview by checking the iteration and best evaluation

        if suppress_output:
            if iter%100 ==0: #print out every 100th iteration update
                print("Iteration")
                print(iter)
                print("Best Eval")
                print(best_eval)
```


### Importing and Exporting Optimizer State

Some optimizer information can be exported or imported. This varies based on each optimizer.

Optimizer state can be exported at any step. When importing an optimizer state, the optimizer should be initialized first, and then the state information can be imported via a Python pickle file. Other methods can be used if custom code is written to handle preprocessing.

Returning data from optimizer and saving to a .pkl file:
```python
    data = demo_optimizer.export_swarm()
    data_df = pd.DataFrame(data)
    print(data_df)
    data_df.to_pickle('output_data_df.pkl')

```

Importing data from a .pkl file and importing it into the optimizer:
```python
    data_df = pd.read_pickle('output_data_df.pkl') 
    demo_optimizer.import_swarm(data_df)

```

### Constraint Handling
Users must create their own constraint function for their problems, if there are constraints beyond the problem bounds. This is then passed into the constructor. If the default constraint function is used, it always returns true (which means there are no constraints).


### Boundary Types
This DE optimizer has 3 different types of bounds, applied per-component: Random (violating components are randomly re-drawn), Reflection (violating components reflect off the bounds), Absorb (violating components are clamped to the bounds).

There is no Invisible option in this optimizer — a trial must be evaluable, since selection needs its fitness.

### Multi-Objective Optimization
The no preference method of multi-objective optimization, but a Pareto Front is not calculated. Instead, the best choice (smallest norm of output vectors) is listed as the output.

### Objective Function Handling

The objective function is handled in two parts. 

* First, a defined function, such as one passed in from `func_F.py` (see examples), is evaluated based on current trial vector locations. This allows for the optimizers to be utilized in the context of 1. benchmark functions from the objective function library, 2. user defined functions, 3. replacing explicitly defined functions with outside calls to programs such as simulations or other scripts that return a matrix of evaluated outputs. 

* Secondly, the actual objective function is evaluated. In the AntennaCAT set of optimizers, the objective function evaluation is either a `TARGET` or `THRESHOLD` evaluation. For a `TARGET` evaluation, which is the default behavior, the optimizer minimizes the absolute value of the difference of the target outputs and the evaluated outputs. A `THRESHOLD` evaluation includes boolean logic to determine if a 'greater than or equal to' or 'less than or equal to' or 'equal to' relation between the target outputs (or thresholds) and the evaluated outputs exist. 

Future versions may include options for function minimization when target values are absent. 


#### Creating a Custom Objective Function

Custom objective functions can be used by creating a directory with the following files:
* configs_F.py
* constr_F.py
* func_F.py

`configs_F.py` contains lower bounds, upper bounds, the number of input variables, the number of output variables, the target values, and a global minimum if known. This file is used primarily for unit testing and evaluation of accuracy. If these values are not known, or are dynamic, then they can be included experimentally in the controller that runs the optimizer's state machine. 

`constr_F.py` contains a function called `constr_F` that takes in an array, `X`, of trial vector positions to determine if the vector is in a valid or invalid location. 

`func_F.py` contains the objective function, `func_F`, which takes two inputs. The first input, `X`, is the array of trial vector positions. The second input, `NO_OF_OUTS`, is the integer number of output variables, which is used to set the array size. In included objective functions, the default value is hardcoded to work with the specific objective function.

Below are examples of the format for these files.

`configs_F.py`:
```python
OBJECTIVE_FUNC = func_F
CONSTR_FUNC = constr_F
OBJECTIVE_FUNC_NAME = "one_dim_x_test.func_F" #format: FUNCTION NAME.FUNCTION
CONSTR_FUNC_NAME = "one_dim_x_test.constr_F" #format: FUNCTION NAME.FUNCTION

# problem dependent variables
LB = [[0]]             # Lower boundaries
UB = [[1]]             # Upper boundaries
IN_VARS = 1            # Number of input variables (x-values)
OUT_VARS = 1           # Number of output variables (y-values) 
TARGETS = [0]          # Target values for output
GLOBAL_MIN = []        # Global minima sample, if they exist. 

```

`constr_F.py`, with no constraints:
```python
def constr_F(x):
    F = True
    return F
```

`constr_F.py`, with constraints:
```python
def constr_F(X):
    F = True
    # objective function/problem constraints
    if (X[2] > X[0]/2) or (X[2] < 0.1):
        F = False
    return F
```

`func_F.py`:
```python
import numpy as np
import time

def func_F(X, NO_OF_OUTS=1):
    F = np.zeros((NO_OF_OUTS))
    noErrors = True
    try:
        x = X[0]
        F = np.sin(5 * x**3) + np.cos(5 * x) * (1 - np.tanh(x ** 2))
    except Exception as e:
        print(e)
        noErrors = False

    return [F], noErrors
```

#### Internal Objective Function Example

There are three functions included in the repository:
1) Himmelblau's function, which takes 2 inputs and has 1 output
2) A multi-objective function with 3 inputs and 2 outputs (see lundquist_3_var)
3) A single-objective function with 1 input and 1 output (see one_dim_x_test)

Each function has four files in a directory:
   1) configs_F.py - contains imports for the objective function and constraints, CONSTANT assignments for functions and labeling, boundary ranges, the number of input variables, the number of output values, and the target values for the output
   2) constr_F.py - contains a function with the problem constraints, both for the function and for error handling in the case of under/overflow. 
   3) func_F.py - contains a function with the objective function.
   4) graph.py - contains a script to graph the function for visualization.

Other multi-objective functions can be applied to this project by following the same format (and several have been collected into a compatible library, and will be released in a separate repo)

<p align="center">
        <img src="media/himmelblau_plots.png" alt="Himmelblau’s function" height="250">
</p>
   <p align="center">Plotted Himmelblau’s Function with 3D Plot on the Left, and a 2D Contour on the Right</p>

```math
f(x, y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2
```

| Global Minima | Boundary | Constraints |
|----------|----------|----------|
| f(3, 2) = 0                 | $-5 \leq x,y \leq 5$  |   | 
| f(-2.805118, 3.121212) = 0  | $-5 \leq x,y \leq 5$  |   | 
| f(-3.779310, -3.283186) = 0 | $-5 \leq x,y \leq 5$  |   | 
| f(3.584428, -1.848126) = 0  | $-5 \leq x,y \leq 5$   |   | 

<p align="center">
        <img src="media/obj_func_pareto.png" alt="Function Feasible Decision Space and Objective Space with Pareto Front" height="200">
</p>
   <p align="center">Plotted Multi-Objective Function Feasible Decision Space and Objective Space with Pareto Front</p>

```math
\text{minimize}: 
\begin{cases}
f_{1}(\mathbf{x}) = (x_1-0.5)^2 + (x_2-0.1)^2 \\
f_{2}(\mathbf{x}) = (x_3-0.2)^4
\end{cases}
```

| Num. Input Variables| Boundary | Constraints |
|----------|----------|----------|
| 3      | $0.21\leq x_1\leq 1$ <br> $0\leq x_2\leq 1$ <br> $0.1 \leq x_3\leq 0.5$  | $x_3\gt \frac{x_1}{2}$ or $x_3\lt 0.1$| 

<p align="center">
        <img src="media/1D_test_plots.png" alt="Function Feasible Decision Space and Objective Space with Pareto Front" height="200">
</p>
   <p align="center">Plotted Single Input, Single-objective Function Feasible Decision Space and Objective Space with Pareto Front</p>

```math
f(\mathbf{x}) = sin(5 * x^3) + cos(5 * x) * (1 - tanh(x^2))
```
| Num. Input Variables| Boundary | Constraints |
|----------|----------|----------|
| 1      | $0\leq x\leq 1$  | $0\leq x\leq 1$| |

Local minima at $(0.444453, -0.0630916)$

Global minima at $(0.974857, -0.954872)$

### Target vs. Threshold Configuration

An April 2025 feature is the user ability to toggle TARGET and THRESHOLD evaluation for the optimized values. The key variables for this are:

```python
# Boolean. use target or threshold. True = THRESHOLD, False = EXACT TARGET
evaluate_threshold = True  

# array
TARGETS = func_configs.TARGETS    # Target values for output from function configs
# OR:
TARGETS = [0,0,0] #manually set BASED ON PROBLEM DIMENSIONS

# threshold is same dims as TARGETS
# 0 = use target value as actual target. value should EQUAL target
# 1 = use as threshold. value should be LESS THAN OR EQUAL to target
# 2 = use as threshold. value should be GREATER THAN OR EQUAL to target
#DEFAULT THRESHOLD
THRESHOLD = np.zeros_like(TARGETS) 
# OR
THRESHOLD = [0,1,2] # can be any mix of TARGET and THRESHOLD  
```

To implement this, the original `self.Flist` objective function calculation has been replaced with the function `objective_function_evaluation`, which returns a numpy array.

The original calculation:
```python
self.Flist = abs(self.targets - self.Fvals)
```
Where `self.Fvals` is a re-arranged and error checked returned value from the passed in function from `func_F.py` (see examples for the internal objective function or creating a custom objective function). 

When using a THRESHOLD, the `Flist` value corresponding to the target is set to epsilon (the smallest system value) if the evaluated `func_F` value meets the threshold condition for that target item. If the threshold is not met, the absolute value of the difference of the target output and the evaluated output is used. With a THRESHOLD configuration, each value in the numpy array is evaluated individually, so some values can be 'greater than or equal to' the target while others are 'equal' or 'less than or equal to' the target. 


## Example Implementations

### Basic DE Example
`main_test.py` provides a sample use case of the optimizer. 

### Detailed Messages
`main_test_details.py` provides an example using a parent class, and the self.suppress_output flag to control error messages that are passed back to the parent class to be printed with a timestamp. This implementation sets up the hooks for integration with AntennaCAT in order to provide the user feedback of warnings and errors.

### Realtime Graph

`main_test_graph.py` provides an example using a parent class, and the self.suppress_output flag to control error messages that are passed back to the parent class to be printed with a timestamp. Additionally, a realtime graph shows trial vector locations at every step.

NOTE: if you close the graph as the code is running, the code will continue to run, but the graph will not re-open.

## References

[1] R. Storn and K. Price, "Differential Evolution – A Simple and Efficient Heuristic for Global Optimization over Continuous Spaces," Journal of Global Optimization, vol. 11, pp. 341–359, 1997, doi: 10.1023/A:1008202821328.

[2] S. Das and P. N. Suganthan, "Differential Evolution: A Survey of the State-of-the-Art," IEEE Transactions on Evolutionary Computation, vol. 15, no. 1, pp. 4–31, 2011, doi: 10.1109/TEVC.2010.2059031.

[3] P. Rocca, G. Oliveri, and A. Massa, "Differential Evolution as Applied to Electromagnetics," IEEE Antennas and Propagation Magazine, vol. 53, no. 1, pp. 38–49, 2011, doi: 10.1109/MAP.2011.5773566.

## Related Publications and Repositories
This software works as a stand-alone implementation, and as one of the optimizers integrated into AntennaCAT.

## Licensing

The code in this repository has been released under GPL-2.0