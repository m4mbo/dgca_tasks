import numpy as np
from grow.reservoir import Reservoir
from util.consts import T


def shannon_entropy(data: list):
    # count the frequency of unique elements
    _, counts = np.unique(data, return_counts=True)
    probs = counts / len(data)  # normalize
    entropy = -np.sum(probs * np.log2(probs))
    return entropy

def kernel_rank(res: Reservoir, 
                input: np.ndarray=None, 
                state: np.ndarray=None, 
                num_timesteps: int=T):
    if input is None:
        input = np.random.uniform(-1, 1, (1, num_timesteps)).astype(np.float64)
        input = np.repeat(input, res.input_units, axis=0)
    if state is None:
        res.reset()
        _ = res.run(input)
    return np.linalg.matrix_rank(res.reservoir_state[:, res.washout:])

def generalization_measure(res: Reservoir, 
                           input: np.ndarray=None, 
                           output: np.ndarray=None,
                           num_timesteps: int=T, 
                           epsilon: float=0.03, 
                           tau: float=1, 
                           num_bins: int=5):
    """
    MGR + TGR
    Griffin, D., Stepney, S., 2024. 
    Entropy Transformation Measures for Computational Capacity
    """
    if input is None:
        input = np.random.uniform(-1, 1, (1, num_timesteps)).astype(np.float64)
        input = np.repeat(input, res.input_units, axis=0)
        res.reset()
        output = res.run(input)

    if np.any(np.isnan(output)):
        return np.nan
    
    # accounting for washout
    input = input[:, res.washout:]
    num_timesteps -= res.washout

    # sampling similar inputs                               
    simh = np.array([], dtype=int)
    for t in range(num_timesteps):
        simh = np.append(simh, sample_history(t, input, num_timesteps, epsilon, tau))

    # computing differences
    diff_in = np.array([], dtype=float)
    diff_out = np.array([], dtype=float)
    for t in range(num_timesteps): 
        diff_in = np.append(diff_in, (1/tau) * np.sum(np.array([np.abs(input[:,t-i] - input[:,simh[t]-i]) for i in range(tau)])))
        diff_out = np.append(diff_out, np.abs(output[:,t] - output[:,simh[t]]))

    return shannon_entropy([exponential_discretization(i, num_bins, np.max(diff_in)) for i in diff_in]) / \
        shannon_entropy([exponential_discretization(o, num_bins, np.max(diff_out)) for o in diff_out])

def exponential_discretization(x: float, num_bins: int, upper_bound: float):
    """
    Helper function for generalization_measure.
    Discretizes a value x into a bin number.
    """  
    if x <= 0 or upper_bound <= 0:
        return 0    
    return np.max((1, np.ceil(np.log2(x/upper_bound) + num_bins)))
            
def sample_history(i: int, 
                   input: np.ndarray,
                   num_timesteps: int, 
                   epsilon: float, 
                   tau: float) -> int:
    """
    Helper function for generalization_measure.
    Samples input.
    """
    choices = np.array([])
    for j in range(num_timesteps):
        h = np.sum(np.array([np.abs(input[:,i-k] - input[:,j-k]) for k in range(tau+1)]))
        if h < epsilon:
            choices = np.append(choices, j)
    return int(np.random.choice(choices))

def get_metrics(res: Reservoir, 
                num_timesteps: int=T, 
                epsilon: float=0.03, 
                tau: float=1, 
                num_bins: int=5):
    """
    Runs the input once and collects both KR and GM.
    """
    input = np.random.uniform(-1, 1, (1, T)).astype(np.float64)
    input = np.repeat(input, res.input_units, axis=0)
    res.reset()
    output = res.run(input)

    if np.any(np.isnan(output)) or np.any(np.isinf(output)):
        return np.nan, np.nan

    return kernel_rank(res, input, res.reservoir_state, num_timesteps), \
        generalization_measure(res, input, output, num_timesteps, epsilon, tau, num_bins)
    
def linear_memory_capacity(res: Reservoir,
                           input: np.ndarray=None,
                           output: np.ndarray=None,
                           num_timesteps: int=T,
                           predictions: np.ndarray=None,
                           filter: float=0.0):
    """
    """
    sequence_length = num_timesteps // 2

    if input is None:
        input = np.random.uniform(-1, 1, (1, num_timesteps)).astype(np.float64)
        input = np.repeat(input, res.input_units, axis=0)

    if output is None:
        output = np.zeros((res.output_units, num_timesteps))
        for i in range(res.output_units):
            if i < num_timesteps:
                output[i, i:num_timesteps] = input[0, 0:num_timesteps - i]
            else:
                output[i, 0:num_timesteps - i] = input[0, i:num_timesteps]  # wrap around index

    # split
    train_input = input[:,:sequence_length]
    train_output = output[:,:sequence_length]

    test_input = input[:,sequence_length:]
    test_output = output[:,sequence_length:]
    test_output = test_output[:,res.washout:]

    # train the reservoir and get predictions if not provided
    if predictions is None:
        res.reset()  # reset the reservoir state
        _ = res.train(train_input, target=train_output) 
        predictions = res.run(test_input)  # run the test input through the reservoir

    # compute the linear memory capacity as the sum of r^2 scores across delays
    
    memory_capacities = []
    for i in range(res.output_units):
        mean_output = np.mean(test_output[i, :])
        mean_predict = np.mean(predictions[i, :])
        sz = predictions.shape[1]

        covariance = np.mean((test_output[i, :] - mean_output) * 
                      (predictions[i, :] - mean_predict) / (sz - 1))
        prediction_variance = np.var(predictions[i, :])
        input_variance = np.var(test_input[i, res.washout:])

        # Memory capacity calculation
        memory_capacity = (covariance ** 2) / (input_variance * prediction_variance)
        if memory_capacity < filter:
            memory_capacity = 0.0
        memory_capacities.append(memory_capacity)

    return np.sum(memory_capacities)
        
    
