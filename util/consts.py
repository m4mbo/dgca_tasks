import numpy as np
from scipy.special import expit

N_STATES = 3

Q_M = [[1,0], 
       [0,0]]

Q_F = [[0,1], 
       [0,0]]

Q_B = [[0,0], 
       [1,0]]

Q_N = [[0,0], 
       [0,1]]

POLAR_TABLE = np.array([
    [1, 1, -1],  # state_from 0
    [-1, 1, 1],  # state_from 1
    [1, -1, 1]   # state_from 2
])

T = 2000

# activation functions
def linear(x):
    return x

def sigmoid(x):
    return expit(x)

def tanh(x):
    return np.tanh(x)

ACTIVATION_TABLE = np.array([tanh, tanh, linear])