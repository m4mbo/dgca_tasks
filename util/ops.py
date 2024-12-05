import numpy as np
from dgca.reservoir import Reservoir

NRMSE = lambda y,y_fit: np.mean(((y-y_fit)**2)/np.var(y))
MAE = lambda y,y_fit: np.mean(np.abs(y, y_fit))

def one_hot(x: np.ndarray):
    """
    Helper function to on hot encode an array x.
    """
    tf = x == np.max(x, axis=0, keepdims=True)
    return tf.astype(int)

def rindex(it, li):
    """
    Reverse index() ie.
    index of last occurence of item in list
    """
    return len(li) - 1 - li[::-1].index(it)