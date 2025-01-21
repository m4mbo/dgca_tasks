import numpy as np
from grow.reservoir import Reservoir


def shannon_entropy(series):
    # count the frequency of unique elements
    _, counts = np.unique(series, return_counts=True)
    probabilities = counts / len(series)  # normalize
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

def kernel_rank(res: Reservoir, input: np.ndarray=None, state: np.ndarray=None, num_timesteps: int=None):
    if not input:
        if not num_timesteps:
            num_timesteps = int(res.size() * 1.5)
        input = np.random.uniform(-1, 1, num_timesteps).astype(np.float64)
    if not state:
        _, state = res.fit(input)
    return np.linalg.matrix_rank(state)

def generalization_measure(res: Reservoir, input: np.ndarray=None, output: np.ndarray=None, num_timesteps: int=None, epsilon: float=0.1, tau: float=1, num_bins: int=5):
    
    if not input:
        if not num_timesteps:
            num_timesteps = int(res.size() * 1.5)
        input = np.random.uniform(-1, 1, num_timesteps).astype(np.float64)
    else:
        num_timesteps = len(input)
                                                            
    simh = np.array([])
    for t in range(num_timesteps):
        simh = np.append(simh, sample_history(t, input, num_timesteps, epsilon, tau))

    diff_in = np.array([])
    diff_out = np.array([])
    for t in range(num_timesteps): 
        diff_in = np.append(diff_in, (1/tau) * np.sum(np.array([np.abs(input[t-i] - input[simh[t]-i]) for i in range(tau-1)])))
        diff_out = np.append(diff_out, np.abs(output[t] - output[simh[t]]))

    return shannon_entropy([exponential_discretization(i, num_bins, np.max(diff_in)) for i in diff_in]) / \
        shannon_entropy([exponential_discretization(o, num_bins, np.max(diff_out)) for o in diff_out])

def exponential_discretization(x: float, num_bins: int, upper_bound: float):
    """
    Helper function for generalization_measure.
    Discretizes a value x into a bin number.
    """
    return np.max(1, np.ceil(np.log2(x/upper_bound) + num_bins))
            
def sample_history(i, input, num_timesteps, epsilon, tau):
    """
    Helper function for generalization_measure.
    Samples input.
    """
    choices = np.array([])
    for j in range(num_timesteps):
        h = np.sum(np.array([np.abs(input[i-k] - input[j-k]) for k in range(tau)]))
        if h < epsilon:
            choices = np.append(choices, j)
    return np.random.choice(choices)
        
def linear_memory_capacity(res: Reservoir):
    pass
