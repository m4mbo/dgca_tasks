#%%
import os
import numpy as np
from dgca.runner import Runner
from dgca.reservoir import Reservoir
from evolve.fitness import NarmaFitness
from evolve.mga import ChromosomalMGA, EvolvableDGCA

def get_seed(n_io, n_states):
    
    if n_io:
        n_nodes = n_io + 1
        A = np.zeros((n_nodes, n_nodes), dtype=int)
        
        # input nodes
        for i in range(n_io // 2):
            A[i, -1] = 1
        # output nodes
        for i in range(n_io // 2, n_io):
            A[-1, i] = 1

        S = np.zeros((n_nodes, n_states), dtype=int)  
        S[:, 0] = 1
    else:
        A = np.array([[0]])
        S = np.zeros((1, n_states), dtype=int)  
        S[0, 0] = 1
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
N_IO = 10

# min_conenctivity
conditions = {'max_size': 300, 
              'min_size': 100, 
              'end2end': True}

fitness_fn = NarmaFitness(conditions=conditions, 
                          verbose=True, 
                          order=ORDER,
                          fixed_seq=True)

A, S = get_seed(N_IO, 3)

reservoir = Reservoir(A=A, S=S, n_io=N_IO)
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