import numpy as np

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

def dfs_directed(A: np.ndarray, current: int, visited: set) -> bool:
    """
    Perform a recursive DFS on a directed adjacency matrix
    """
    # current node as visited
    visited.add(current)
    
    # visit neighbors
    neighbors = np.nonzero(A[current])[0]  # directed neighbors
    for neighbor in neighbors:
        if neighbor not in visited:
            if dfs_directed(A, neighbor, visited):
                return True
    
    return False