import numpy as np

Q_M = [[1,0], 
       [0,0]]

Q_F = [[0,1], 
       [0,0]]

Q_B = [[0,0], 
       [1,0]]

Q_N = [[0,0], 
       [0,1]]

POLAR_TABLE = np.array([
    [1,  1, -1],  # state_from = 0
    [-1, 1,  1],  # state_from = 1
    [1, -1, 1]    # state_from = 2
])