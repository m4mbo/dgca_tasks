import numpy as np
from grow.reservoir import Reservoir, check_conditions
from tasks.series import *
from tasks.metrics import kernel_rank, generalization_measure

NRMSE = lambda y,y_fit: np.mean(((y-y_fit)**2)/np.var(y))

class ReservoirFitness:
    """
    Interface for a Callable which takes a graph and returns its fitnesss
    """
    def __init__(self, high_good: bool):
        self.high_good = high_good

    def __call__(self, res: Reservoir):
        raise NotImplementedError
    
class TaskFitness(ReservoirFitness):

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
            self.input, self.target = series(order=self.order)

    def __call__(self, res: Reservoir) -> float:
        
        checks_ok = check_conditions(res, self.conditions, self.verbose)
        
        res_ = res.copy()
        if not self.self_loops:
            res_ = res.no_selfloops()
        
        if checks_ok:   
            if not self.fixed_series:
                self.input, self.target = self.series(order=self.order)
            res_.reset()
            predictions = res_.bipolar().train(self.input, target=self.target)
            err = np.min((NRMSE(self.target[:, res.washout:], predictions), 1))     # normalized root mean square error
            if self.verbose:
                print(f'Skipped {self.skip_count}')
            self.skip_count = 0
        else:
            self.skip_count += 1
            err = np.nan
        return err
    
class MetricFitness(ReservoirFitness):

    def __init__(self,
                 conditions: dict = dict(),
                 self_loops: bool= False,
                 verbose: bool = False,
                 metric: str = 'KR') -> None:
        
        super().__init__(high_good=True) # because we will be returning an error metric, so low is good.
        self.self_loops = self_loops
        self.conditions = conditions
        self.verbose = verbose
        self.skip_count = 0
        self.memo = {'fitness':[], 'graph':[], 'model':[]}
        self.metric = metric

        if metric == 'KR':
            self.metric_fn = kernel_rank
        elif metric == 'GM':
            self.metric_fn = generalization_measure
        else:
            raise ValueError('metric must be either "KR" or "GM"')
        
    def __call__(self, res: Reservoir) -> float:
        
        checks_ok = check_conditions(res, self.conditions, self.verbose)
        res_ = res.copy()

        if not self.self_loops:
            res_ = res.no_selfloops()

        if checks_ok:   
            err = self.metric_fn(res_.bipolar())
            if self.metric == "KR":  
                size = res_.size()
                err = (1 - err / size) * (np.log(size) + 1)
            if self.verbose:
                print(f'Skipped {self.skip_count}')
            self.skip_count = 0
        else:
            self.skip_count += 1
            err = np.nan
        return err