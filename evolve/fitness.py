import numpy as np
from grow.reservoir import Reservoir, check_conditions
from tasks.series import narma

NRMSE = lambda y,y_fit: np.mean(((y-y_fit)**2)/np.var(y))

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
                 verbose: bool = False,
                 fixed_seq: bool = True
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
        self.fixed_seq = fixed_seq
        if fixed_seq:
            self.u, self.y = narma(2000, self.order)

    def __call__(self, res: Reservoir) -> float:
        
        checks_ok = check_conditions(res, self.conditions, self.verbose)
        
        if checks_ok:   
            if not self.fixed_seq:
                self.u, self.y = narma(2000, self.order)
            w_out, state = res.bipolar().fit(self.u, self.input_gain, self.feedback_gain, y_train=self.y)
            y_fit = w_out.T @ state
            err = NRMSE(self.y, y_fit)     # normalized root mean square error
            if self.verbose:
                print(f'Skipped {self.skip_count}')
            self.skip_count = 0
        else:
            self.skip_count += 1
            err = np.nan
        return err