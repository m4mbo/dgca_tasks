#%%
import os
import numpy as np
from dgca.runner import Runner
from dgca.reservoir import Reservoir
from evolve.fitness import NarmaFitness
from evolve.mga import ChromosomalMGA, EvolvableDGCA

def get_seed(n_fixed, n_states):
    
    n_nodes = n_fixed + 1
    A = np.zeros((n_nodes, n_nodes), dtype=int)
    
    # input nodes
    for i in range(n_fixed // 2):
        A[i, -1] = 1
    # output nodes
    for i in range(n_fixed // 2, n_fixed):
        A[-1, i] = 1

    S = np.zeros((n_nodes, n_states), dtype=int)  
    S[:, 0] = 1
    return A, S

#%%
PROJ_DIR = os.path.dirname(os.path.abspath(__file__))

# various settings
POPULATION_SIZE = 30
MUTATE_RATE = 0.02
CROSS_RATE = 0.5
CROSS_STYLE = 'cols'
NUM_TRIALS = 5000
ORDER = 10
N_FIXED = 8

# min_conenctivity
conditions = {'min_size': N_FIXED+20,
              'end_to_end': True}

fitness_fn = NarmaFitness(conditions=conditions, 
                          verbose=False, 
                          order=ORDER)

A, S = get_seed(N_FIXED, 3)

reservoir = Reservoir(A=A, S=S, n_fixed=N_FIXED)
model = EvolvableDGCA(n_states=reservoir.n_states)  
runner = Runner(max_steps=100, max_size=300)
mga = ChromosomalMGA(popsize=POPULATION_SIZE,
                     seed_graph=reservoir,
                     model=model,
                     runner=runner,
                     fitness_fn=fitness_fn,
                     mutate_rate=MUTATE_RATE,
                     cross_rate=CROSS_RATE,
                     cross_style=CROSS_STYLE,
                     parquet_filename="fitness.parquet"
                     )
#%%
mga.run(steps=NUM_TRIALS)

# %%