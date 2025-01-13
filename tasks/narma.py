import numpy as np
import sklearn.linear_model as linfit

def narma_sequence(t, order=10):
    """
    Creates NARMA-X sequence for t timesteps, where X is the order of the system.
    """
    x = order - 1
    # input
    u = np.random.uniform(0, 0.5, t).astype(np.float64)
    # NARMA sequence
    y = np.zeros(t)
    for i in range(t-1):
        sum_t = np.sum(y[0:i]) if i <= x else np.sum(y[i-x:i+1])
        if i <= x:
            y[i+1] = 0.3 * y[i] + 0.05 * sum_t * y[i] + 0.1
        else:
            y[i+1] = 0.3 * y[i] + 0.05 * sum_t * y[i] + 1.5 * u[i] * u[i-x] + 0.1
        # cap to prevent overflows
        y[i+1] = np.clip(y[i+1], -1e10, 1e10)


    # discard transient effects from first 20 steps
    u = u[np.newaxis, 20:]
    y = y[np.newaxis, 20:]

    return u, y

def fit_model(u, w_in, w_res, inputgain, feedbackgain, n_io=None, y_train=None):
    """
    Fits a reservoir computing model using Bayesian Ridge Regression.

    Parameters:
    - u (input_dim, time_steps): Input data.
    - w_in (input_dim, reservoir_dim): Input weight matrix.
    - w_res (reservoir_dim, reservoir_dim): Reservoir weight matrix, the ESN.
    - inputgain: Scaling factor for w_in.
    - feedbackgain: Scaling factor for w_res.
    - n_ouput: Number of fixed I/O nodes. Assuming output nodes are the second half of those nodes.
    - y_train (1, time_steps): Target data.
    
    Returns:
    - w_out (reservoir_dim + 1, 1): Output weight matrix.
    - reservoir_state (reservoir_dim + 1, time_steps): State matrix.
    """
    reservoir_state = np.zeros((w_res.shape[0], u.shape[1]))
    for i in range(u.shape[1]):
        if i == 0:
            reservoir_state[:,i] = np.tanh(inputgain*w_in.T @ u[:,i])
        else:
            reservoir_state[:,i] = np.tanh(inputgain*w_in.T @ u[:,i] + feedbackgain*w_res.T @ reservoir_state[:,i-1])
    
    if n_io:
        reservoir_state = reservoir_state[n_io//2:n_io] # keeping only output nodes
    
    # add bias node
    reservoir_state = np.concatenate((reservoir_state, np.ones((1, reservoir_state.shape[1]))),axis=0)

    w_out = None
    if y_train is not None:
        # Tikhonov regularization to fit and generalize on unseen data
        regression = linfit.BayesianRidge(max_iter=3000, tol=1e-6, verbose=False, fit_intercept=False)
        # remove first element as it does not consider previous state
        try:
            regression.fit(reservoir_state[:,1:].T, y_train[0,1:])
        except ValueError as e:
            print(y_train[0,1:])
        w_out = regression.coef_
    
    return w_out, reservoir_state
    




