import numpy as np
from dgca.reservoir import Reservoir
from tasks.narma import narma_sequence, fit_model
from util.ops import NRMSE

class GraphFitness:
    """
    Interface for a Callable which takes a graph and returns its fitnesss
    """
    def __init__(self, high_good: bool):
        self.high_good = high_good

    def __call__(self, res: Reservoir):
        raise NotImplementedError
    
class NarmaFitness(GraphFitness):

    def __init__(self,
                 conditions: dict = dict(),
                 self_loops: bool= False,
                 order: int=10,
                 input_gain: float=0.1,
                 feedback_gain: float=0.95,
                 verbose: bool = False
                 ) -> None:
        super().__init__(high_good = False) # because we will be returning an error metric, so low is good.
        self.self_loops = self_loops
        self.conditions = conditions
        self.verbose = verbose
        self.order = order
        self.input_gain = input_gain
        self.feedback_gain = feedback_gain
        self.skip_count = 0
        self.memo = {'fitness':[], 'graph':[], 'model':[]}

    def __call__(self, res: Reservoir) -> float:
        
        checks_ok = res.check_conditions(self.conditions, self.verbose)
        
        if checks_ok:   
            u, y = narma_sequence(2000, self.order)
            w_in = np.random.randint(-1, 2, (1, res.size()))
            w_in[:, res.n_fixed//2:] = 0  # masking all nodes except input
            w_out, state = fit_model(u, w_in, res.A, self.input_gain, self.feedback_gain, y_train=y, n_fixed=res.n_fixed)
            y_fit = w_out.T @ state
            err = NRMSE(y, y_fit)     # normalized root mean square error
            if self.verbose:
                print(f'Skipped {self.skip_count}')
            self.skip_count = 0
        else:
            self.skip_count += 1
            err = np.nan
        return err