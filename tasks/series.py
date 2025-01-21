import numpy as np

def narma(t, order=10):
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


    




