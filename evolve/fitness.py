import numpy as np
from grow.reservoir import Reservoir, check_conditions
from tasks.series import *

NRMSE = lambda y,y_fit: np.mean(((y-y_fit)**2)/np.var(y))

class GraphFitness:
    """
    Interface for a Callable which takes a graph and returns its fitnesss
    """
    def __init__(self, high_good: bool):
        self.high_good = high_good

    def __call__(self, res: Reservoir):
        raise NotImplementedError
    
class TaskFitness(GraphFitness):

    def __init__(self,
                 series: callable,
                 conditions: dict = dict(),
                 self_loops: bool= False,
                 order: int=10,
                 verbose: bool = False,
                 fixed_series: bool = True
                 ) -> None:
        super().__init__(high_good = False) # because we will be returning an error metric, so low is good.
        self.self_loops = self_loops
        self.conditions = conditions
        self.verbose = verbose
        self.order = order
        self.skip_count = 0
        self.series = series
        self.memo = {'fitness':[], 'graph':[], 'model':[]}
        self.fixed_series = fixed_series
        if fixed_series:
            self.u, self.y = series(2000, self.order)

    def __call__(self, res: Reservoir) -> float:
        
        checks_ok = check_conditions(res, self.conditions, self.verbose)
        
        if checks_ok:   
            if not self.fixed_series:
                self.u, self.y = self.series(2000, self.order)
            w_out, state = res.bipolar().fit(self.u, y_train=self.y)
            y_fit = w_out.T @ state
            err = NRMSE(self.y, y_fit)     # normalized root mean square error
            if self.verbose:
                print(f'Skipped {self.skip_count}')
            self.skip_count = 0
        else:
            self.skip_count += 1
            err = np.nan
        return err
    
class MetricFitness():
    pass