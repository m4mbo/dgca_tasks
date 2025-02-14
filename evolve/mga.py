from tqdm import tqdm
import numpy as np
import jsonpickle
from grow.dgca import DGCA, MLP
from grow.reservoir import Reservoir
from evolve.fitness import ReservoirFitness
from grow.runner import Runner
import pandas as pd


class Chromosome:
    """
    Data for a "chromosome" (MLP weights and biases). 
    Implements mutation and crossover methods.
    """
    def __init__(self, 
                 weights: list[np.ndarray], 
                 biases: list[np.ndarray], 
                 mutate_rate: float, 
                 crossover_rate: float, 
                 crossover_style: str, 
                 best_fitness: np.float32 = np.nan):
        self.weights = weights  # list of weight matrices 
        self.biases = biases    # list of bias vectors 
        self.mutate_rate = mutate_rate
        self.crossover_rate = crossover_rate
        self.crossover_style = crossover_style
        self.best_fitness = best_fitness

    def mutate(self, rate: float = None) -> "Chromosome":
        """
        Mutates all weight and bias matrices in the MLP.
        """
        if rate is None:
            rate = self.mutate_rate
        for i, w in enumerate(self.weights):
            mask = np.random.choice([True, False], size=w.shape, p=[rate, 1 - rate])
            random_vals = np.random.uniform(-1, 1, size=w.shape)
            self.weights[i][mask] = random_vals[mask]

        for i, b in enumerate(self.biases):
            mask = np.random.choice([True, False], size=b.shape, p=[rate, 1 - rate])
            random_vals = np.random.uniform(-1, 1, size=b.shape)
            self.biases[i][mask] = random_vals[mask]

        self.best_fitness = np.nan
        return self

    def crossover(self, other: "Chromosome") -> "Chromosome":
        """
        Crosses over weight and bias matrices with another chromosome.
        """
        for i in range(len(self.weights)):
            if self.crossover_style == 'rows':
                row_idx = np.random.choice(
                    range(self.weights[i].shape[0]), 
                    size=int(self.weights[i].shape[0] * self.crossover_rate), 
                    replace=False
                )
                self.weights[i][row_idx, :] = other.weights[i][row_idx, :]
            elif self.crossover_style == 'cols':
                col_idx = np.random.choice(
                    range(self.weights[i].shape[1]), 
                    size=int(self.weights[i].shape[1] * self.crossover_rate), 
                    replace=False
                )
                self.weights[i][:, col_idx] = other.weights[i][:, col_idx]

        for i in range(len(self.biases)):
            idx = np.random.choice(
                range(self.biases[i].shape[0]), 
                size=int(self.biases[i].shape[0] * self.crossover_rate), 
                replace=False
            )
            self.biases[i][idx] = other.biases[i][idx]

        self.best_fitness = np.nan
        return self

    def get_new(self) -> "Chromosome":
        """
        Creates a new ChromosomeMLP with random weights and biases.
        """
        new_weights = [np.random.uniform(-1, 1, w.shape) for w in self.weights]
        new_biases = [np.random.uniform(-1, 1, b.shape) for b in self.biases]
        return Chromosome(new_weights, new_biases, self.mutate_rate, self.crossover_rate, self.crossover_style)

    def crossover(self, other: "Chromosome") -> "Chromosome":
        """
        Crosses over weight and bias matrices with another chromosome.

        Parameters:
        - other: Chromosome, the other parent Chromosome.

        Returns:
        - Chromosome: self, with crossed-over data.
        """
        for i in range(len(self.weights)):
            if self.crossover_style == 'rows':
                # Row-wise crossover
                row_idx = np.random.choice(
                    range(self.weights[i].shape[0]),
                    size=int(self.weights[i].shape[0] * self.crossover_rate),
                    replace=False
                )
                # Swap rows
                self.weights[i][row_idx, :] = other.weights[i][row_idx, :]
            elif self.crossover_style == 'cols':
                # Column-wise crossover
                col_idx = np.random.choice(
                    range(self.weights[i].shape[1]),
                    size=int(self.weights[i].shape[1] * self.crossover_rate),
                    replace=False
                )
                # Swap columns
                self.weights[i][:, col_idx] = other.weights[i][:, col_idx]

        for i in range(len(self.biases)):
            # Bias crossover (element-wise)
            idx = np.random.choice(
                range(self.biases[i].shape[0]),
                size=int(self.biases[i].shape[0] * self.crossover_rate),
                replace=False
            )
            self.biases[i][idx] = other.biases[i][idx]

        self.best_fitness = np.nan
        return self

    def get_new(self) -> "Chromosome":
        """
        Creates a new Chromosome with random weights and biases.
        """
        new_weights = [np.random.uniform(-1, 1, w.shape) for w in self.weights]
        new_biases = [np.random.uniform(-1, 1, b.shape) for b in self.biases]
        return Chromosome(new_weights, new_biases, self.mutate_rate, self.crossover_rate, self.crossover_style)
   

class EvolvableDGCA(DGCA):
    def __init__(self, n_states, hidden_size=64):
        super().__init__(n_states=n_states, hidden_size=hidden_size)

    def set_chromosomes(self, chr_action: Chromosome, chr_state: Chromosome):
        self.action_mlp.set_parameters(chr_action.weights, chr_action.biases)
        self.state_mlp.set_parameters(chr_state.weights, chr_state.biases)

    def get_chromosomes(self, mutate_rate: float, cross_rate: float, cross_style: str):
        weights_action, biases_action = self.action_mlp.get_parameters()
        weights_state, biases_state = self.state_mlp.get_parameters()
        chr_action = Chromosome(weights_action, biases_action, mutate_rate, cross_rate, cross_style)
        chr_state = Chromosome(weights_state, biases_state, mutate_rate, cross_rate, cross_style)
        return chr_action, chr_state


class ChromosomalMGA:

    def __init__(self, 
                 popsize: int,
                 model: EvolvableDGCA,
                 seed_graph: Reservoir,
                 runner: Runner,
                 fitness_fn: ReservoirFitness,
                 mutate_rate: float, 
                 cross_rate: float, 
                 cross_style: str,
                 exp_id: int,
                 parquet_filename: str | None = None):
        self.popsize = popsize
        self.model = model
        self.seed_graph = seed_graph
        self.runner = runner
        self.fitness_fn = fitness_fn
        self.parquet_filename = parquet_filename
        self.exp_id = exp_id

        # nan tolerant
        if self.fitness_fn.high_good:
            self.better = lambda a, b: np.isnan(b) or a>=b
        else:
            self.better = lambda a, b: np.isnan(b) or a<=b

        self.base_chromosomes = self.model.get_chromosomes(mutate_rate, cross_rate, cross_style)
        self.pop_chromosomes = np.array([[bc.get_new() for _ in range(self.popsize)] for bc in self.base_chromosomes]).T
        self.num_chromosomes = len(self.base_chromosomes)
        self.fitness_record = [] 
        if self.parquet_filename is not None:
            print(f'Results will be written to: {self.parquet_filename}')

    def run(self, steps: int) -> list[float]:
        pbar = tqdm(range(steps),postfix={'fit':0,'best':0})
        for _ in pbar:
            f = self.contest()
            best_fitness = np.max(self.fitness_record) if self.fitness_fn.high_good else np.min(self.fitness_record)
            pbar.set_postfix({'fit':f,'best':best_fitness})

    def contest(self) -> float:
        """
        Runs a single contest between two individuals created out of randomly selected chromosomes
        Returns the fitness of the fitter one.
        """
        # select two sets of chromosomes at random
        idx = np.random.randint(low=0,high=self.popsize,size=(2,self.num_chromosomes))
        contestant_chromosomes = np.take_along_axis(self.pop_chromosomes, idx, axis=0)
        fitness = (np.nan, np.nan)
        
        # keep looping until they are not nan, doesn't count as new 'runs'
        while np.all(np.isnan(fitness)):
            fitness = self.run_individual(contestant_chromosomes[0]), self.run_individual(contestant_chromosomes[1])
            # if both contestants' fitness are nan, mutate the chromosomes
            if np.all(np.isnan(fitness)):
                for i, chr in enumerate(contestant_chromosomes.flat):
                    if np.isnan(fitness[i // self.num_chromosomes]):
                        if np.isnan(chr.best_fitness):
                            chr.mutate(rate=1.0)  # do 100% mutation in this case
                        else:
                            chr.mutate()  # default mutation rate
        
        win, lose = (0,1) if self.better(*fitness) else (1,0)
        for c in range(self.num_chromosomes):
            if idx[win,c]==idx[lose,c]:
                # chromosomes were the same, don't change anything
                continue
            else:
                chr_win, chr_lose = contestant_chromosomes[win,c], contestant_chromosomes[lose,c]
                # don't change the losing individual if it has previously been part of an 
                # individual with higher fitness than the winner
                if self.better(fitness[win], chr_lose.best_fitness):
                    # call crossover & mutate on the loser (this changes it in place)
                    chr_lose.crossover(chr_win).mutate()
        return fitness[win]
    
    def log_fitness(self, fitness: float, reservoir: Reservoir):
        new_row = {
            "exp_id": self.exp_id,
            "epoch": len(self.fitness_record),
            "fitness": fitness,
            "model": jsonpickle.encode(self.model),
            "final_reservoir": jsonpickle.encode(reservoir),
            "skip_count": self.fitness_fn.skip_count
        }
        new_data = pd.DataFrame([new_row])
        try:
            existing_data = pd.read_parquet(self.parquet_filename)
            updated_data = pd.concat([existing_data, new_data], ignore_index=True)
        except FileNotFoundError:
            updated_data = new_data
        updated_data.to_parquet(self.parquet_filename, index=False)

    def run_individual(self, chromosomes: list[Chromosome]) -> float:
        """
        Returns the fitness of one set of chromosomes.
        """
        self.model.set_chromosomes(*chromosomes)
        self.runner.reset()
        final_res = self.runner.run(self.model, self.seed_graph)
        fitness = self.fitness_fn(final_res)
        # update chromosomes' best_fitness score
        for chr in chromosomes:
            if self.better(fitness, chr.best_fitness):
                chr.best_fitness = fitness
        if not(np.isnan(fitness)) and (self.parquet_filename is not None):
            self.fitness_record.append(fitness)
            self.log_fitness(fitness, final_res)
        return fitness