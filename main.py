#%%
import os
import numpy as np
from grow.runner import Runner
from grow.reservoir import Reservoir
from evolve.fitness import NarmaFitness
from evolve.mga import ChromosomalMGA, EvolvableDGCA

def get_seed(input_nodes, output_nodes, n_states):
    
    if input_nodes or output_nodes:
        n_nodes = input_nodes + output_nodes + 1
        A = np.zeros((n_nodes, n_nodes), dtype=int)
        
        # input nodes
        for i in range(input_nodes):
            A[i, -1] = 1
        # output nodes
        for i in range(input_nodes, input_nodes+output_nodes):
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
CROSS_RATE = 0.6
CROSS_STYLE = 'cols'
NUM_TRIALS = 5000
ORDER = 10
INPUT = 0
OUTPUT = 0

# min_conenctivity
conditions = {'max_size': 300, 
              'min_size': 100
              }

fitness_fn = NarmaFitness(conditions=conditions, 
                          verbose=True, 
                          order=ORDER,
                          fixed_seq=True)

A, S = get_seed(INPUT, OUTPUT, 3)

reservoir = Reservoir(A=A, S=S, input_nodes=INPUT, output_nodes=OUTPUT)
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