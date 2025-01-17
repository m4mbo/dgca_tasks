from tqdm import tqdm
import numpy as np
import jsonpickle
from grow.dgca import DGCA
from grow.reservoir import Reservoir
from evolve.fitness import GraphFitness
from grow.runner import Runner
import pandas as pd


class Chromosome:
    """
    Data for a "chromosome" (SLP weights). 
    Implements mutation and crossover methods.
    """

    def __init__(self, 
                 data: np.ndarray, 
                 mutate_rate: float, 
                 crossover_rate: float, 
                 crossover_style: str, 
                 best_fitness: np.float32 = np.nan):
        """
        Constructor to initialize a Chromosome.

        Parameters:
        - data: np.ndarray, the weights or information stored in the chromosome.
        - mutate_rate: float, the rate at which mutation occurs.
        - crossover_rate: float, the rate at which crossover occurs.
        - crossover_style: str, style of crossover ('rows' or 'cols').
        - best_fitness: np.float32, fitness score, default is NaN.
        """
        self.data = data
        self.mutate_rate = mutate_rate
        self.crossover_rate = crossover_rate
        self.crossover_style = crossover_style
        self.best_fitness = best_fitness

    def mutate(self, rate: float=None) -> "Chromosome":
        """
        Mutate the chromosome's data.

        Parameters:
        - rate: Optional float, overrides the default mutation rate if provided.

        Returns:
        - Chromosome: self, with mutated data.
        """
        if rate is None:
            rate = self.mutate_rate
        # mutation mask based on mutation rate
        mask = np.random.choice([True, False], p=[rate, 1 - rate], size=self.data.shape)
        # apply mutations (replace selected elements with random values)
        random = np.random.uniform(size=self.data.shape, low=-1, high=1).astype(np.float32)
        self.data[mask] = random[mask]
        self.best_fitness = np.nan
        return self

    def crossover(self, other: "Chromosome") -> "Chromosome":
        """
        Perform crossover between two Chromosomes.

        Parameters:
        - other: Chromosome, the other parent Chromosome.

        Returns:
        - Chromosome: self, with crossed-over data.
        """
        assert self.data.shape == other.data.shape, "Mismatched shapes for Chromosome.data"
        n_rows, n_cols = self.data.shape
        if self.crossover_style == 'rows':
            # randomly select rows to swap with other chromosome
            row_idx = np.random.choice(range(n_rows), 
                                       size=(round(n_rows * self.crossover_rate),), 
                                       replace=False)
            self.data[row_idx, :] = other.data[row_idx, :]
        elif self.crossover_style == 'cols':
            # randomly select columns to swap with other chromosome
            col_idx = np.random.choice(range(n_cols), 
                                       size=(round(n_cols * self.crossover_rate),), 
                                       replace=False)
            self.data[:, col_idx] = other.data[:, col_idx]
        else:
            raise ValueError("crossover_style should be 'rows' or 'cols'")
        self.best_fitness = np.nan
        return self

    def get_new(self):
        """
        Factory method to create a new Chromosome with the same attributes but new random data.

        Returns:
        - Chromosome: A new Chromosome instance.
        """
        newdata = np.random.uniform(size=self.data.shape, low=-1, high=1).astype(np.float32)
        return Chromosome(newdata, self.mutate_rate, self.crossover_rate, self.crossover_style, best_fitness=np.nan)
   
class EvolvableDGCA(DGCA):

    def set_chromosomes(self, chr1: Chromosome, chr2: Chromosome) -> None:
        self.w_action = chr1.data
        self.w_state = chr2.data

    def get_chromosomes(self, mutate_rate: float, cross_rate: float, cross_style: str) -> list[Chromosome]:
        chr1 = Chromosome(self.w_action.copy(), mutate_rate, cross_rate, cross_style)
        chr2 = Chromosome(self.w_state.copy(), mutate_rate, cross_rate, cross_style)
        return chr1, chr2

class ChromosomalMGA:

    def __init__(self, 
                 popsize: int,
                 model: EvolvableDGCA,
                 seed_graph: Reservoir,
                 runner: Runner,
                 fitness_fn: GraphFitness,
                 mutate_rate: float, 
                 cross_rate: float, 
                 cross_style: str,
                 parquet_filename: str | None = None):
        self.popsize = popsize
        self.model = model
        self.seed_graph = seed_graph
        self.runner = runner
        self.fitness_fn = fitness_fn
        self.parquet_filename = parquet_filename
        
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
            "fitness": fitness,
            "model": jsonpickle.encode(self.model),
            "final_graph": jsonpickle.encode(reservoir),
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
        final_graph = self.runner.run(self.model, self.seed_graph)
        fitness = self.fitness_fn(final_graph)
        # update chromosomes' best_fitness score
        for chr in chromosomes:
            if self.better(fitness, chr.best_fitness):
                chr.best_fitness = fitness
        if not(np.isnan(fitness)) and (self.parquet_filename is not None):
            self.fitness_record.append(fitness)
            self.log_fitness(fitness, final_graph)
        return fitness